"""Thread-based job queue with pause / resume / cancel.

The queue knows nothing about Qt: the UI subscribes with callbacks and
marshals them onto the GUI thread itself.
"""
from __future__ import annotations

import enum
import itertools
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import log


class JobState(str, enum.Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    PAUSED = "Paused"
    DONE = "Done"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class Cancelled(Exception):
    """Raised inside a job when the user presses Stop."""


class JobControl:
    """Passed to every job function: progress reporting + pause/cancel checks."""

    def __init__(self, job: "Job", queue: "JobQueue"):
        self.job = job
        self._queue = queue
        self._proc = None  # running subprocess, so pause/cancel can reach it

    def progress(self, fraction: float, message: str = "", eta: Optional[float] = None):
        self.job.progress = max(0.0, min(1.0, fraction))
        if message:
            self.job.message = message
        self.job.eta = eta
        self._queue._emit("progress", self.job)

    def check(self):
        """Block while paused; raise Cancelled if stopped."""
        if self.job.cancel_event.is_set():
            raise Cancelled()
        while not self.job.resume_event.is_set():
            if self.job.cancel_event.wait(0.2):
                raise Cancelled()
        if self.job.cancel_event.is_set():
            raise Cancelled()

    @property
    def cancelled(self) -> bool:
        return self.job.cancel_event.is_set()

    def attach_process(self, proc):
        self._proc = proc
        self.job._proc = proc
        if not self.job.resume_event.is_set():
            self._queue._suspend(proc)

    def detach_process(self):
        self._proc = None
        self.job._proc = None


@dataclass
class Job:
    kind: str                     # "photo" | "video" | "preview"
    source: str
    func: Callable[["JobControl"], Any]
    title: str = ""
    id: int = 0
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    message: str = ""
    eta: Optional[float] = None
    result: Any = None
    error: str = ""
    started: float = 0.0
    finished: float = 0.0
    options: Dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    resume_event: threading.Event = field(default_factory=threading.Event)
    _proc: Any = None

    def __post_init__(self):
        self.resume_event.set()
        if not self.title:
            self.title = self.source


class JobQueue:
    """Runs jobs one at a time (the GPU is the bottleneck) in a worker thread."""

    _ids = itertools.count(1)

    def __init__(self, workers: int = 1):
        self.jobs: List[Job] = []
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._listeners: Dict[str, List[Callable]] = {}
        self._paused_all = False
        self._stopping = False
        self._threads = [threading.Thread(target=self._loop, daemon=True, name=f"job-worker-{i}")
                         for i in range(workers)]
        for t in self._threads:
            t.start()

    # ---------------- events ----------------
    def on(self, event: str, cb: Callable):
        """Events: added, started, progress, finished, changed."""
        self._listeners.setdefault(event, []).append(cb)

    def _emit(self, event: str, job: Optional[Job]):
        for cb in self._listeners.get(event, []) + self._listeners.get("changed", []) * (event != "changed"):
            try:
                cb(job)
            except Exception:
                log.exception("listener failed")

    # ---------------- public API ----------------
    def add(self, kind: str, source: str, func: Callable, title: str = "", **options) -> Job:
        job = Job(kind=kind, source=source, func=func, title=title, id=next(self._ids), options=options)
        with self._wake:
            if self._paused_all:
                job.resume_event.clear()
            self.jobs.append(job)
            self._wake.notify_all()
        self._emit("added", job)
        return job

    def get(self, job_id: int) -> Optional[Job]:
        with self._lock:
            return next((j for j in self.jobs if j.id == job_id), None)

    def pause(self, job: Optional[Job] = None):
        with self._lock:
            targets = [job] if job else [j for j in self.jobs if j.state in (JobState.QUEUED, JobState.RUNNING)]
            if job is None:
                self._paused_all = True
            for j in targets:
                j.resume_event.clear()
                if j.state == JobState.RUNNING:
                    j.state = JobState.PAUSED
                    if j._proc is not None:
                        self._suspend(j._proc)
                self._emit("changed", j)

    def resume(self, job: Optional[Job] = None):
        with self._wake:
            targets = [job] if job else list(self.jobs)
            if job is None:
                self._paused_all = False
            for j in targets:
                if j.state in (JobState.PAUSED, JobState.QUEUED, JobState.RUNNING):
                    j.resume_event.set()
                    if j.state == JobState.PAUSED:
                        j.state = JobState.RUNNING
                        if j._proc is not None:
                            self._resume_proc(j._proc)
                    self._emit("changed", j)
            self._wake.notify_all()

    def cancel(self, job: Optional[Job] = None):
        with self._wake:
            targets = [job] if job else [j for j in self.jobs if j.state in
                                         (JobState.QUEUED, JobState.RUNNING, JobState.PAUSED)]
            for j in targets:
                j.cancel_event.set()
                j.resume_event.set()
                if j._proc is not None:
                    self._resume_proc(j._proc)
                    self._kill(j._proc)
                if j.state == JobState.QUEUED:
                    j.state = JobState.CANCELLED
                self._emit("changed", j)
            self._wake.notify_all()

    def retry(self, job: Job) -> Optional[Job]:
        if job.state not in (JobState.FAILED, JobState.CANCELLED):
            return None
        with self._wake:
            job.state = JobState.QUEUED
            job.progress, job.error, job.message = 0.0, "", "Retrying (resumes where it stopped)"
            job.cancel_event.clear()
            job.resume_event.set()
            self._wake.notify_all()
        self._emit("changed", job)
        return job

    def remove_finished(self):
        with self._lock:
            self.jobs = [j for j in self.jobs if j.state not in
                         (JobState.DONE, JobState.FAILED, JobState.CANCELLED)]
        self._emit("changed", None)

    def move(self, job: Job, delta: int):
        with self._lock:
            i = self.jobs.index(job)
            k = max(0, min(len(self.jobs) - 1, i + delta))
            self.jobs.insert(k, self.jobs.pop(i))
        self._emit("changed", None)

    def pending(self) -> List[Job]:
        with self._lock:
            return [j for j in self.jobs if j.state in (JobState.QUEUED, JobState.RUNNING, JobState.PAUSED)]

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        end = None if timeout is None else time.time() + timeout
        while self.pending():
            if end is not None and time.time() > end:
                return False
            time.sleep(0.05)
        return True

    def shutdown(self):
        self.cancel()
        with self._wake:
            self._stopping = True
            self._wake.notify_all()

    @property
    def paused(self) -> bool:
        return self._paused_all

    # ---------------- worker ----------------
    def _next(self) -> Optional[Job]:
        for j in self.jobs:
            if j.state == JobState.QUEUED and j.resume_event.is_set():
                return j
        return None

    def _loop(self):
        while True:
            with self._wake:
                job = self._next()
                while job is None and not self._stopping:
                    self._wake.wait(0.5)
                    job = self._next()
                if self._stopping:
                    return
                job.state = JobState.RUNNING
                job.started = time.time()
            self._emit("started", job)
            ctl = JobControl(job, self)
            try:
                ctl.check()
                job.result = job.func(ctl)
                job.state = JobState.DONE
                job.progress = 1.0
                job.message = job.message or "Finished"
            except Cancelled:
                job.state = JobState.CANCELLED
                job.message = "Cancelled by user"
            except Exception as e:
                job.state = JobState.FAILED
                job.error = friendly_error(e)
                job.message = job.error
                log.error("Job %s failed: %s\n%s", job.title, e, traceback.format_exc())
            finally:
                ctl.detach_process()
                job.finished = time.time()
                job.eta = None
            self._emit("finished", job)

    # ---------------- process control ----------------
    @staticmethod
    def _suspend(proc):
        try:
            import psutil
            p = psutil.Process(proc.pid)
            for c in p.children(recursive=True):
                c.suspend()
            p.suspend()
        except Exception:
            pass

    @staticmethod
    def _resume_proc(proc):
        try:
            import psutil
            p = psutil.Process(proc.pid)
            p.resume()
            for c in p.children(recursive=True):
                c.resume()
        except Exception:
            pass

    @staticmethod
    def _kill(proc):
        try:
            import psutil
            p = psutil.Process(proc.pid)
            for c in p.children(recursive=True):
                c.kill()
            p.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def friendly_error(e: BaseException) -> str:
    """Turn exceptions into plain-English messages."""
    text = str(e) or e.__class__.__name__
    low = text.lower()
    if isinstance(e, MemoryError) or "out of memory" in low or "vkallocatememory" in low:
        return ("Your graphics card ran out of memory. Go to Settings and lower the Tile size "
                "(try 128 or 64), then press Retry.")
    if "vkcreateinstance failed" in low or "invalid gpu device" in low or "no vulkan" in low:
        return ("No compatible graphics driver was found (Vulkan). Update your GPU driver from "
                "NVIDIA, AMD or Intel, or pick 'CPU' in Settings (much slower).")
    if isinstance(e, PermissionError):
        return f"Windows blocked access to a file or folder: {getattr(e, 'filename', '') or text}. Pick another output folder."
    if isinstance(e, FileNotFoundError):
        return f"A file could not be found: {getattr(e, 'filename', '') or text}"
    if "no space left" in low or "disk full" in low or "errno 28" in low:
        return "The disk is full. Free some space or choose another temp/output folder in Settings."
    return text
