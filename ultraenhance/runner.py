"""Run a native tool as a child process with progress, pause and cancel."""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, List, Optional

from . import engines
from .config import log
from .job_queue import Cancelled, JobControl

_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)%")


class EngineStalled(RuntimeError):
    """The child process stopped making progress and was stopped by the watchdog."""


def run_tool(cmd: List[str], ctl: Optional[JobControl] = None,
             on_line: Optional[Callable[[str], None]] = None, name: str = "tool",
             on_tick: Optional[Callable[[], None]] = None,
             progress_counter: Optional[Callable[[], int]] = None,
             stall_seconds: float = 0) -> List[str]:
    """on_tick runs about once a second even when the tool prints nothing, so the
    progress bar keeps moving. If progress_counter and stall_seconds are given and
    the counter does not grow for stall_seconds (time spent paused not counted),
    the tool is stopped and EngineStalled is raised instead of waiting forever."""
    log.info("Running %s: %s", name, " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    proc = engines.popen(cmd)
    if ctl:
        ctl.attach_process(proc)
    tail: deque[str] = deque(maxlen=40)

    def reader():
        buf = ""
        assert proc.stdout is not None
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    tail.append(buf.strip())
                    if on_line:
                        try:
                            on_line(buf.strip())
                        except Exception:
                            pass
                buf = ""
            else:
                buf += ch
        if buf.strip():
            tail.append(buf.strip())

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    last_tick = 0.0
    last_count = progress_counter() if progress_counter else 0
    last_change = time.monotonic()
    try:
        while proc.poll() is None:
            t.join(0.25)
            if ctl and ctl.cancelled:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise Cancelled()
            now = time.monotonic()
            paused = bool(ctl and not ctl.job.resume_event.is_set())
            if paused:
                last_change = now
                continue
            if on_tick and now - last_tick >= 1.0:
                last_tick = now
                try:
                    on_tick()
                except Exception:
                    pass
            if progress_counter and stall_seconds > 0:
                try:
                    c = progress_counter()
                except Exception:
                    c = last_count
                if c != last_count:
                    last_count, last_change = c, now
                elif now - last_change > stall_seconds:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    t.join(2)
                    log.error("%s made no progress for %.0f s; stopped. Last output:\n%s",
                              name, stall_seconds, "\n".join(tail))
                    raise EngineStalled(f"{name} stopped responding (no progress for "
                                        f"{int(stall_seconds // 60)} minutes).")
        t.join(2)
    finally:
        if ctl:
            ctl.detach_process()
    if ctl and ctl.cancelled:
        raise Cancelled()
    if proc.returncode != 0:
        detail = "\n".join(list(tail)[-8:])
        log.error("%s failed (%s):\n%s", name, proc.returncode, "\n".join(tail))
        raise RuntimeError(f"{name} stopped with an error.\n{detail}")
    return list(tail)


def percent_from_line(line: str) -> Optional[float]:
    m = _PCT.search(line)
    if not m:
        return None
    v = float(m.group(1))
    return v / 100 if 0 <= v <= 100 else None
