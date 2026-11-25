"""Video enhancement pipeline: decode -> (interpolate) -> AI upscale in chunks -> encode -> mux.

Resumable: every job has a folder under %USERPROFILE%\\UltraEnhancePro\\jobs
holding a state.json. Finished chunks are kept as encoded segments, so an
interrupted job restarts at the first unfinished chunk instead of frame 1.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional

from . import engines, face_restore, ffmpeg_utils as fu
from . import frame_dedupe as dd
from .config import JOBS_DIR, log
from .estimates import (PNG_BYTES_PER_PIXEL, free_bytes, human_bytes, human_time,
                        video_output_bytes, video_seconds, video_temp_bytes_streamed)
from .formats import VIDEO_CONTAINERS
from .job_queue import Cancelled, JobControl
from .photo import EngineContext, unique_output_path
from .presets import MODELS, compute_geometry, plan_passes, required_factor
from .runner import EngineStalled, percent_from_line, run_tool

PNG_PATTERN = "%08d.png"
# How many chunks of source frames the background decoder may run ahead of the AI.
STREAM_AHEAD_CHUNKS = 2


@dataclass
class VideoOptions:
    preset: str = "1080p Full HD"
    fit_mode: str = "fit"
    model: str = "Fast (recommended)"
    interpolate: bool = False
    target_fps: int = 60
    deinterlace: str = "auto"      # auto | on | off
    denoise: bool = False
    stabilize: bool = False
    codec: str = "H.264"
    encoder_family: str = "auto"   # auto | nvenc | amf | qsv | software
    quality: str = "Balanced"
    crf: Optional[int] = None
    bitrate_kbps: Optional[int] = None
    container: str = "MP4"
    output_dir: str = ""
    chunk_frames: int = 0          # 0 = automatic
    # --- enhancement extras (added in 1.x; old saved jobs simply get the defaults)
    skip_duplicates: bool = True   # upscale each run of identical frames only once
    face_enhance: bool = False     # GFPGAN face restore on the upscaled frames
    face_strength: float = 0.7     # a little below photos: less frame-to-frame flicker
    cleanup: str = "off"           # off | light | strong  (deblock + denoise old/compressed video)
    auto_color: bool = False       # mild automatic levels + saturation
    sharpen: bool = False          # mild unsharp mask after the final resize


# Options added after the first release, with their defaults. A job whose new
# options are still at these defaults keeps the job folder it had before the
# options existed, so jobs started with an older version can still be resumed.
_LATER_OPTIONS = {"skip_duplicates": True, "face_enhance": False, "face_strength": 0.7,
                  "cleanup": "off", "auto_color": False, "sharpen": False}


def options_from_dict(data: Optional[dict]) -> VideoOptions:
    """Build VideoOptions from saved settings, ignoring unknown keys and repairing bad values."""
    names = {f.name for f in fields(VideoOptions)}
    o = VideoOptions(**{k: v for k, v in (data or {}).items() if k in names})
    if o.cleanup not in fu.CLEANUP_LEVELS:
        o.cleanup = "off"
    if o.model not in MODELS:
        o.model = VideoOptions.model
    return o


def _job_key_options(opts: VideoOptions) -> dict:
    d = asdict(opts) | {"output_dir": ""}
    for k, default in _LATER_OPTIONS.items():
        if d.get(k) == default:
            d.pop(k, None)
    return d


@dataclass
class VideoPlan:
    info: fu.MediaInfo
    geo: object
    passes: List[int]
    ai_w: int
    ai_h: int
    out_fps: Fraction
    frames_in: int
    frames_out: int
    temp_bytes: int
    output_bytes: int
    seconds: float
    warnings: List[str]
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def summary(self) -> str:
        g = self.geo
        lines = [
            f"Source: {self.info.width}x{self.info.height}, {float(self.info.fps):.3f} fps, "
            f"{human_time(self.info.duration)}, {self.frames_in} frames",
            f"Output: {g.canvas_w}x{g.canvas_h} at {float(self.out_fps):.3f} fps ({self.frames_out} frames)",
            f"AI passes: {' + '.join(f'{p}x' for p in self.passes) or 'none (resize only)'}",
            f"Temporary disk space needed: about {human_bytes(self.temp_bytes)}"
            + (" (frames are processed a chunk at a time and deleted as they go)"
               if self.frames_out == self.frames_in else " (60 fps smoothing keeps every frame on disk)"),
            f"Estimated output file size: about {human_bytes(self.output_bytes)}",
            f"Estimated time: about {human_time(self.seconds)} (real time depends on your GPU)",
        ]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines + [f"Warning: {w}" for w in self.warnings])


def make_plan(src: str, opts: VideoOptions, ctx: EngineContext, info: Optional[fu.MediaInfo] = None,
              duration: Optional[float] = None) -> VideoPlan:
    info = info or fu.probe(str(engines.ffprobe_path()), src)
    geo = compute_geometry(info.width, info.height, opts.preset, None, opts.fit_mode, even_dims=True)
    _, native = MODELS[opts.model]
    factor = required_factor(info.width, info.height, geo)
    passes = plan_passes(factor, native)
    # Video: a second full AI pass costs ~16x more time. When one pass already gets
    # within 1.5x of the target, finish with a high-quality Lanczos resize instead.
    if len(passes) > 1 and passes[0] * 1.5 >= factor:
        passes = passes[:1]
    prod = 1
    for p in passes:
        prod *= p
    ai_w, ai_h = info.width * prod, info.height * prod
    dur = duration if duration is not None else info.duration
    frames_in = max(1, int(round(dur * info.fps_float))) if dur else max(1, info.frames)
    out_fps = info.fps
    frames_out = frames_in
    if opts.interpolate and info.fps_float and opts.target_fps > info.fps_float + 0.5:
        out_fps = Fraction(opts.target_fps)
        frames_out = int(round(frames_in * opts.target_fps / info.fps_float))
    chunk = chunk_size(opts, ai_w, ai_h)
    codec = fu.effective_codec(opts.codec, opts.container, geo.canvas_w, geo.canvas_h)
    out_bytes = video_output_bytes(geo.canvas_w, geo.canvas_h, dur, codec, opts.quality,
                                   float(out_fps), opts.bitrate_kbps)
    temp = temp_bytes_needed(info.width, info.height, ai_w, ai_h, frames_in, frames_out, chunk,
                             len(passes), out_bytes, streamed=frames_out == frames_in)
    secs = (video_seconds(ai_w, ai_h, frames_out, 1, ctx.cpu, 1.0, MODELS[opts.model][0])
            + (frames_in * 0.03 if opts.interpolate else 0))
    warnings, notes = [], []
    if opts.face_enhance:
        secs += frames_out * face_seconds_per_frame(ctx.cpu)
        if face_restore.available():
            notes.append("Face enhancement is on: faces are restored on every unique frame (adds time).")
        else:
            warnings.append("Face enhancement models are missing, so faces will not be enhanced.")
    if opts.skip_duplicates and passes:
        notes.append("Duplicate frames are upscaled only once, so videos with still or slow "
                     "scenes finish faster than this estimate.")
    if opts.cleanup != "off" or opts.auto_color or opts.sharpen:
        extras = ([f"{opts.cleanup} artefact cleanup"] if opts.cleanup != "off" else []) + \
                 (["auto colour"] if opts.auto_color else []) + (["sharpening"] if opts.sharpen else [])
        notes.append("Restoration filters: " + ", ".join(extras) + ".")
    if codec != opts.codec and opts.container != "WEBM":
        warnings.append(f"H.264 cannot encode wider than {fu.H264_MAX_SIDE} pixels, so H.265/HEVC "
                        "will be used for this size.")
    if geo.canvas_h >= 4320 or geo.canvas_w >= 7680:
        warnings.append("8K video is extremely slow and the file will be very large. "
                        "Try the 5-second preview first.")
    elif min(geo.canvas_w, geo.canvas_h) >= 3240:
        warnings.append("6K video is very slow and the file will be large. "
                        "Try the 5-second preview first.")
    elif geo.canvas_h >= 2160 or geo.canvas_w >= 3840:
        warnings.append("4K video takes a long time. A 5-second preview is recommended first.")
    if ctx.cpu:
        warnings.append("No GPU selected: CPU mode is many times slower.")
    if opts.container == "WEBM" and opts.codec != "AV1":
        warnings.append("WEBM only supports AV1 here, so AV1 will be used.")
    return VideoPlan(info, geo, passes, ai_w, ai_h, out_fps, frames_in, frames_out,
                     int(temp), int(out_bytes), secs, warnings, notes)


def temp_bytes_needed(src_w: int, src_h: int, ai_w: int, ai_h: int, frames_in: int, frames_out: int,
                      chunk: int, passes: int, output_bytes: int, streamed: bool = True) -> int:
    """Peak temporary disk space for a job.

    Without frame interpolation, frames are decoded only a little ahead of the
    AI (STREAM_AHEAD_CHUNKS chunks) and deleted after each chunk is encoded, so
    the need does not grow with the length of the video beyond the encoded
    segments. RIFE needs every source frame at once, so with interpolation all
    source + interpolated frames are counted."""
    if streamed:
        return video_temp_bytes_streamed(src_w, src_h, ai_w, ai_h, chunk, STREAM_AHEAD_CHUNKS * chunk,
                                         output_bytes, passes)
    png = PNG_BYTES_PER_PIXEL
    return int((src_w * src_h * png * (frames_in + frames_out)
                + ai_w * ai_h * png * chunk * (2 if passes > 1 else 1) + output_bytes) * 1.1)


def face_seconds_per_frame(cpu: bool = False) -> float:
    """Rough GFPGAN cost per frame (detection + one or two faces)."""
    return 1.2 if cpu else 0.35


def chunk_size(opts: VideoOptions, ai_w: int, ai_h: int) -> int:
    if opts.chunk_frames:
        return opts.chunk_frames
    mpix = ai_w * ai_h / 1e6
    # ~1.5 GB of upscaled PNGs per chunk at most, 24..240 frames.
    return int(max(24, min(240, 1500 / max(1.0, mpix * 1.6))))


def job_dir_for(src: str, opts: VideoOptions, start: Optional[float], duration: Optional[float]) -> Path:
    st = os.stat(src)
    key = json.dumps([os.path.abspath(src).lower(), st.st_size, int(st.st_mtime),
                      _job_key_options(opts), start, duration], sort_keys=True, default=str)
    return JOBS_DIR / hashlib.sha1(key.encode()).hexdigest()[:16]


class _State:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"stage": "new", "done_chunks": [], "frames": 0}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v
        self.save()


def _count_frames(d: Path) -> int:
    return sum(1 for _ in d.glob("*.png"))


# Watchdog: if the AI engine writes no new frame for this long, it is treated as stuck.
STALL_SECONDS_GPU = 20 * 60
STALL_SECONDS_CPU = 90 * 60
MIN_RETRY_TILE = 32


def _link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class FaceStage:
    """GFPGAN face restore for the upscaled frames of a video.

    Missing models or a broken onnxruntime never stop the video: the stage
    turns itself off with a warning in the log and the job carries on."""

    def __init__(self, strength: float = 0.7):
        self.strength = strength
        self.faces_found = 0
        self.ran = False
        try:
            self.enabled = face_restore.available()
        except Exception:  # pragma: no cover - defensive
            self.enabled = False
        if not self.enabled:
            log.warning("Face enhancement is on, but the face models are missing; "
                        "continuing without face enhancement.")

    def run(self, d: Path, n: int, ctl: Optional[JobControl] = None,
            on_frame: Optional[Callable[[int], None]] = None) -> int:
        """Restore faces in-place in d/00000001.png .. d/{n}.png. Returns faces restored."""
        check = ctl.check if ctl else None
        found = 0
        for k in range(n):
            if not self.enabled:
                break
            if check:
                check()
            path = d / PNG_PATTERN.replace("%08d", f"{k + 1:08d}")
            try:
                img = dd.read_image(path)
                if img is not None:
                    out, nf = face_restore.restore_faces(img, self.strength, check)
                    if nf:
                        # write via temp + replace: the file may be a hard link to a source frame
                        dd.write_image(path, out)
                        found += nf
                self.ran = True
            except Cancelled:
                raise
            except Exception as e:
                self.enabled = False
                log.warning("Face enhancement turned off for the rest of this video: %s", e)
                break
            if on_frame:
                try:
                    on_frame(k + 1)
                except Cancelled:
                    raise
                except Exception:
                    pass
        self.faces_found += found
        return found


def frame_name(n: int) -> str:
    return f"{n:08d}.png"


def chunk_bounds(ci: int, chunk: int) -> tuple:
    """1-based first and last frame number of chunk ci."""
    return ci * chunk + 1, (ci + 1) * chunk


def first_unfinished_chunk(done: set, seg_exists: Callable[[int], bool]) -> int:
    """Index of the first chunk that still has to be made (decoding resumes there)."""
    ci = 0
    while ci in done and seg_exists(ci):
        ci += 1
    return ci


def decoder_should_wait(next_needed: int, newest_written: int, ahead: int) -> bool:
    """Throttle: pause the decoder once it is `ahead` frames past the oldest frame still in use."""
    return newest_written >= next_needed + ahead


def normalize_frame_names(d: Path) -> int:
    """Make sure the PNGs in d are named 00000001.png .. N.png with no gaps (in sorted order)."""
    files = sorted(d.glob("*.png"))
    if all(f.name == frame_name(i + 1) for i, f in enumerate(files)):
        return len(files)
    tmp = [f.with_name(f"_tmp_{i + 1:08d}.png") for i, f in enumerate(files)]
    for f, t in zip(files, tmp):
        os.replace(f, t)
    for i, t in enumerate(tmp):
        os.replace(t, d / frame_name(i + 1))
    return len(files)


def partition_fstype(path: str, partitions) -> str:
    """File-system type of the partition holding path (longest matching mount point)."""
    p = os.path.normcase(os.path.abspath(str(path)))
    best, fstype = -1, ""
    for part in partitions:
        mp = os.path.normcase(os.path.abspath(part.mountpoint))
        if (p == mp or p.startswith(mp.rstrip("\\/") + os.sep)) and len(mp) > best:
            best, fstype = len(mp), part.fstype
    return fstype


def max_file_bytes(path: str | Path) -> Optional[int]:
    """Largest file the drive holding path can store (None = no practical limit / unknown)."""
    try:
        import psutil
        return fu.fs_file_limit(partition_fstype(str(path), psutil.disk_partitions(all=False)))
    except Exception:
        return None


def _psutil_call(proc, method: str):
    try:
        import psutil
        getattr(psutil.Process(proc.pid), method)()
        return True
    except Exception:
        return False


class StaticFrames:
    """Frames that are all on disk already (after RIFE)."""

    def __init__(self, d: Path, total: int):
        self.dir, self.total, self.finished = d, total, True

    def path(self, n: int) -> Path:
        return self.dir / frame_name(n)

    def wait_for(self, last: int, on_wait=None) -> int:
        return min(last, self.total)

    def release(self, upto: int):
        pass

    def stop(self):
        pass


class FrameStreamer:
    """Decodes the source to PNG frames in the background, only a few chunks ahead.

    The decoder is suspended while it is `ahead` frames past the oldest frame
    still in use and resumed when the AI catches up, so temporary disk use stays
    bounded however long the video is. Running one continuous decode (instead
    of seeking per chunk) keeps temporal filters (denoise, stabilise, auto
    colour) seamless across chunks. A frame counts as complete once the next
    file exists or the decoder has exited.
    """
    POLL = 0.25            # wait_for loop
    MONITOR_POLL = 0.05    # throttle check: a fast decoder overshoots by at most ~this much time
    STALL_SECONDS = 30 * 60

    def __init__(self, cmd: List[str], out_dir: Path, first_frame: int, ahead: int,
                 ctl: Optional[JobControl] = None, name: str = "FFmpeg decode"):
        self.cmd, self.dir, self.first, self.ahead, self.ctl, self.name = cmd, out_dir, first_frame, ahead, ctl, name
        self.window = first_frame          # oldest frame still needed
        self.newest = first_frame - 1      # newest frame file seen on disk
        self.finished = False
        self.total: Optional[int] = None
        self.proc = None
        self._suspended = False
        self._stop = False
        self._tail: deque = deque(maxlen=30)
        self._lock = threading.Lock()

    def path(self, n: int) -> Path:
        return self.dir / frame_name(n)

    def start(self):
        log.info("Running %s (streamed, from frame %d): %s", self.name, self.first, " ".join(map(str, self.cmd)))
        self.proc = engines.popen(self.cmd)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._monitor, daemon=True).start()
        return self

    def _read(self):
        assert self.proc and self.proc.stdout is not None
        buf = ""
        while True:
            ch = self.proc.stdout.read(1)
            if not ch:
                break
            if ch in "\r\n":
                if buf.strip():
                    self._tail.append(buf.strip())
                buf = ""
            else:
                buf += ch

    def _scan(self) -> int:
        # frames before the window may already be used and deleted
        self.newest = max(self.newest, self.window - 1)
        while self.path(self.newest + 1).exists():
            self.newest += 1
        return self.newest

    def _set_suspended(self, want: bool):
        with self._lock:
            if want != self._suspended and self.proc and self.proc.poll() is None:
                if _psutil_call(self.proc, "suspend" if want else "resume"):
                    self._suspended = want

    def _monitor(self):
        while not self._stop and self.proc and self.proc.poll() is None:
            try:
                user_paused = bool(self.ctl and not self.ctl.job.resume_event.is_set())
                self._set_suspended(user_paused or decoder_should_wait(self.window, self._scan(), self.ahead))
            except Exception:
                pass
            time.sleep(self.MONITOR_POLL)

    def wait_for(self, last: int, on_wait: Optional[Callable[[], None]] = None) -> int:
        """Block until frames up to `last` are complete or the decoder ended.
        Returns the newest complete frame number <= last (first-1 if none)."""
        last_seen, last_change, last_tick = self.newest, time.monotonic(), 0.0
        while True:
            if self.ctl:
                self.ctl.check()
            if self.path(last + 1).exists():
                return last
            rc = self.proc.poll() if self.proc else 0
            if rc is not None:
                time.sleep(0.1)
                if rc != 0 and not self._stop:
                    detail = "\n".join(list(self._tail)[-8:])
                    log.error("%s failed (%s):\n%s", self.name, rc, "\n".join(self._tail))
                    raise RuntimeError(f"{self.name} stopped with an error.\n{detail}")
                self.finished = True
                self.total = self._scan()
                return min(last, self.total)
            now = time.monotonic()
            newest = self._scan()
            if newest != last_seen or self._suspended:
                last_seen, last_change = newest, now
            elif now - last_change > self.STALL_SECONDS:
                self.stop()
                raise RuntimeError("Reading the video got stuck (no new frame for 30 minutes). "
                                   "Press Retry; finished parts are kept.")
            if on_wait and now - last_tick >= 1.0:
                last_tick = now
                try:
                    on_wait()
                except Cancelled:
                    raise
                except Exception:
                    pass
            time.sleep(self.POLL)

    def release(self, upto: int):
        """Frames up to `upto` are no longer needed; the decoder may move on."""
        self.window = max(self.window, upto + 1)

    def stop(self):
        self._stop = True
        if self.proc and self.proc.poll() is None:
            self._set_suspended(False)
            try:
                self.proc.kill()
            except Exception:
                pass


def _encoder_for(codec: str, family: str, preferred: Optional[str] = None) -> str:
    available = fu.working_encoders(str(engines.ffmpeg_path()))
    if preferred and preferred in available:
        enc = preferred  # resumed job: keep the encoder its finished segments used
    else:
        enc = fu.choose_encoder(codec, available, family)
    log.info("Encoder chosen: %s (codec %s)", enc, codec)
    return enc


def _check_disk(plan: "VideoPlan", temp_dir: Path, seg_dir: Path, out_dir: Path, container_ext: str):
    """Fail early, with the real numbers, instead of hours into the job."""
    have_segments = sum(f.stat().st_size for f in seg_dir.glob("*.mkv")) if seg_dir.exists() else 0
    need_temp = max(0, plan.temp_bytes - have_segments)
    # +faststart rewrites the finished MP4/MOV once, so the output drive needs room for two copies.
    need_out = int(plan.output_bytes * (2.0 if container_ext in (".mp4", ".mov") else 1.1))
    same_drive = Path(temp_dir).resolve().anchor.lower() == Path(out_dir).resolve().anchor.lower()
    try:
        same_drive = same_drive and os.stat(temp_dir).st_dev == os.stat(out_dir).st_dev
    except OSError:
        pass
    checks = [(temp_dir, need_temp + (need_out if same_drive else 0))] if same_drive else \
        [(temp_dir, need_temp), (out_dir, need_out)]
    for path, need in checks:
        free = free_bytes(path)
        if free < need:
            raise RuntimeError(
                f"Not enough free disk space on {Path(path).anchor or path}: this video needs about "
                f"{human_bytes(need)} there (temporary frames about {human_bytes(need_temp)}, "
                f"finished file about {human_bytes(plan.output_bytes)}), but only {human_bytes(free)} is free. "
                "Free up space, or choose another temp or output folder in Settings.")
    limit = max_file_bytes(out_dir)
    if limit and plan.output_bytes > limit * 0.9:
        raise RuntimeError(
            f"The output drive {Path(out_dir).anchor or out_dir} uses FAT32, which cannot store files "
            f"bigger than 4 GB, and this video will be about {human_bytes(plan.output_bytes)}. Choose an output "
            "folder on an NTFS or exFAT drive in Settings, or pick a lower resolution/quality.")
    log.info("Disk check: temp needs ~%s, output ~%s", human_bytes(need_temp), human_bytes(need_out))


def enhance_video(src: str, opts: VideoOptions, ctx: EngineContext, ctl: Optional[JobControl] = None,
                  start: Optional[float] = None, duration: Optional[float] = None,
                  out_path: Optional[str] = None, keep_temp: bool = False) -> str:
    ffmpeg = str(engines.ffmpeg_path())
    plan = make_plan(src, opts, ctx, duration=duration)
    info, geo = plan.info, plan.geo
    log.info("Video plan for %s:\n%s", src, plan.summary())

    jd = job_dir_for(src, opts, start, duration)
    jd.mkdir(parents=True, exist_ok=True)
    work_root = ctx.temp_dir / ("job_" + jd.name)
    work_root.mkdir(parents=True, exist_ok=True)
    state = _State(jd / "state.json")
    state.data["source"] = src
    state.data["work"] = str(work_root)
    state.data["options"] = asdict(opts)
    state.data["preview"] = start is not None or duration is not None
    state.save()

    in_dir, interp_dir, seg_dir = work_root / "in", work_root / "interp", work_root / "segments"
    for d in (in_dir, interp_dir, seg_dir):
        d.mkdir(exist_ok=True)

    ext = VIDEO_CONTAINERS[opts.container]
    out_dir = Path(opts.output_dir) if opts.output_dir else Path(src).parent / "UltraEnhance"
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_disk(plan, ctx.temp_dir, seg_dir, out_dir, ext)

    t_start = time.time()

    def prog(p, m, eta=None):
        if ctl:
            ctl.check()
            ctl.progress(p, m, eta)

    # -------- 1. decode settings (optional stabilise / deinterlace / cleanup / denoise / colour)
    trf = None
    if opts.stabilize:
        trf_path = work_root / "transforms.trf"
        if not (state.data.get("trf_done") and trf_path.exists()):
            prog(0.01, "Analysing camera shake (stabilise pass 1/2)")
            run_tool(fu.vidstab_detect_cmd(ffmpeg, src, str(trf_path), start, duration), ctl, name="FFmpeg vidstab")
            state["trf_done"] = True
        trf = str(trf_path)
    deint = opts.deinterlace == "on" or (opts.deinterlace == "auto" and info.interlaced)
    filters = fu.pre_filter_chain(deint, opts.denoise, trf, opts.cleanup, opts.auto_color)

    def decode_cmd(skip: int = 0):
        return fu.extract_frames_cmd(ffmpeg, src, str(in_dir / PNG_PATTERN), info.fps, filters,
                                     start, duration, skip_frames=skip)

    chunk = chunk_size(opts, plan.ai_w, plan.ai_h)
    interpolating = bool(opts.interpolate and plan.out_fps != info.fps)
    fps = plan.out_fps if interpolating else info.fps
    p0 = 0.18 if interpolating else 0.03   # where the upscale part of the progress bar starts
    span = 0.93 - p0

    # -------- 2. (60 fps smoothing only) decode everything + RIFE, then drop the source frames
    if interpolating:
        if state["stage"] in ("new", "extracting"):
            state["stage"] = "extracting"
            for f in in_dir.glob("*.png"):
                f.unlink()
            total_in = plan.frames_in

            def on_dec(line):
                n = fu.parse_progress_frame(line)
                if n is not None and ctl:
                    ctl.progress(0.02 + 0.08 * min(1, n / total_in), f"Extracting frames {n}/{total_in}")
            run_tool(decode_cmd(), ctl, on_dec, name="FFmpeg decode")
            n = normalize_frame_names(in_dir)
            if n == 0:
                raise RuntimeError("No frames could be read from this video.")
            state.data["frames"] = n
            state["stage"] = "interpolating"
        if state["stage"] == "interpolating":
            target_frames = int(round(state["frames"] * float(plan.out_fps) / info.fps_float))
            for f in interp_dir.glob("*.png"):
                f.unlink()

            def on_rife():
                if ctl:
                    made = min(_count_frames(interp_dir), target_frames)
                    ctl.progress(0.1 + 0.08 * made / max(1, target_frames),
                                 f"Creating smooth {plan.out_fps} fps frames {made}/{target_frames}")
            prog(0.1, f"Creating smooth {plan.out_fps} fps frames (RIFE)")
            try:
                run_tool(engines.rife_cmd(in_dir, interp_dir, target_frames, ctx.engine_gpu, ctx.threads,
                                          uhd=info.height > 1080), ctl, name="RIFE", on_tick=on_rife,
                         progress_counter=lambda: _count_frames(interp_dir),
                         stall_seconds=STALL_SECONDS_CPU if ctx.cpu else STALL_SECONDS_GPU)
            except EngineStalled:
                raise RuntimeError(
                    "Creating 60 fps frames got stuck on your graphics card. Update your graphics driver, "
                    "or turn off 60 fps smoothing for this video, then press Retry.")
            n = normalize_frame_names(interp_dir)
            if n == 0:
                raise RuntimeError("Frame interpolation produced no frames.")
            state.data["interp_frames"] = n
            state["stage"] = "upscaling"
            for f in in_dir.glob("*.png"):   # source frames are not needed any more
                f.unlink()
        total = int(state.data.get("interp_frames") or _count_frames(interp_dir))
        if total == 0:
            raise RuntimeError("No frames were found to upscale.")
        state.data["frames_total"] = total
    else:
        if state["stage"] != "upscaling":
            state["stage"] = "upscaling"
        total = int(state.data.get("frames_total") or 0)

    # -------- 3. AI upscale + encode, chunk by chunk (resumable, frames deleted as they are used)
    codec = fu.effective_codec(opts.codec, opts.container, geo.canvas_w, geo.canvas_h)
    if codec != opts.codec and opts.container != "WEBM":
        log.info("%s cannot encode %dx%d; using %s instead", opts.codec, geo.canvas_w, geo.canvas_h, codec)
    encoder = _encoder_for(codec, opts.encoder_family, state.data.get("encoder"))
    state["encoder"] = encoder
    scale_vf = fu.scale_filter(geo.canvas_w, geo.canvas_h, geo.scaled_w, geo.scaled_h, geo.offset_x, geo.offset_y,
                               sharpen=opts.sharpen)
    model_name, _ = MODELS[opts.model]
    done = set(state["done_chunks"])

    def seg_path(i: int) -> Path:
        return seg_dir / f"seg_{i:05d}.mkv"

    first_todo = first_unfinished_chunk(done, lambda i: seg_path(i).exists())
    if interpolating:
        source = StaticFrames(interp_dir, total)
    elif total and first_todo * chunk >= total:
        source = StaticFrames(in_dir, total)   # everything already encoded; only the mux is left
    else:
        for f in in_dir.glob("*.png"):          # half-written frames from an interrupted run
            try:
                f.unlink()
            except OSError:
                pass
        source = FrameStreamer(decode_cmd(first_todo * chunk), in_dir, first_todo * chunk + 1,
                               STREAM_AHEAD_CHUNKS * chunk, ctl).start()
    est_total = total or plan.frames_in
    done_frames = sum(min(chunk, max(0, est_total - i * chunk)) for i in done)
    rate_t0, rate_f0 = time.time(), done_frames
    # chunk index (as str, JSON keys) -> duplicate frames that reused an upscaled frame
    skipped = {k: v for k, v in dict(state.data.get("skipped", {})).items() if int(k) in done}
    # Skipping only pays off when there is AI work (upscale or faces) to skip.
    dedupe = opts.skip_duplicates and bool(plan.passes or opts.face_enhance)
    faces = FaceStage(opts.face_strength) if opts.face_enhance else None
    if done:
        log.info("Resuming: %d chunks already finished, continuing at chunk %d", len(done), first_todo + 1)

    def delete_frames(a: int, b: int):
        for n in range(a, b + 1):
            try:
                source.path(n).unlink()
            except OSError:
                pass

    n_chunks = 0
    ci = 0
    try:
        while True:
            a, b = chunk_bounds(ci, chunk)
            if total and a > total:
                break
            seg = seg_path(ci)
            if ci in done and seg.exists():
                if isinstance(source, FrameStreamer) and a >= source.first:
                    # a finished chunk after a gap: its frames are decoded again; just drop them
                    end = source.wait_for(b)
                    delete_frames(a, end)
                    source.release(end)
                    if end < a:
                        break
                    n_chunks = ci + 1
                    if end < b:
                        break
                else:
                    n_chunks = ci + 1
                ci += 1
                continue

            def on_wait():
                if ctl:
                    ctl.progress(p0 + span * min(1.0, done_frames / est_total),
                                 f"Chunk {ci + 1}: reading frames from the video")
            end = source.wait_for(b, on_wait)
            if isinstance(source, FrameStreamer) and source.finished:
                total = est_total = source.total
                state["frames_total"] = total
            if end < a:
                if ci == 0:
                    raise RuntimeError("No frames could be read from this video.")
                break
            files = [source.path(n) for n in range(a, end + 1)]
            missing = [f for f in files if not f.exists()]
            if missing:
                raise RuntimeError("Some frames of this video are missing from the temporary folder. "
                                   "Discard the job on the Home tab and start it again.")
            n_files = len(files)
            n_chunks = ci + 1
            est_total = max(est_total, end)
            chunk_label = f"Chunk {ci + 1}/{-(-est_total // chunk)}"
            prog(p0 + span * min(1.0, done_frames / est_total), chunk_label)
            cin, cout, cfull = work_root / "chunk_in", work_root / "chunk_out", work_root / "chunk_full"
            for d in (cin, cout, cfull):
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir()

            # 3a. find duplicate / near-static frames; only unique frames go to the AI
            if dedupe:
                if ctl:
                    ctl.progress(p0 + span * min(1.0, done_frames / est_total),
                                 f"{chunk_label}: looking for repeated frames")
                ref = dd.find_duplicates(files, check=ctl.check if ctl else None)
            else:
                ref = list(range(n_files))
            uniq = dd.unique_indices(ref)
            n_unique = len(uniq)
            n_dup = n_files - n_unique
            if n_dup:
                log.info("%s: %d of %d frames repeat the previous frame; upscaling %d unique frames",
                         chunk_label, n_dup, n_files, n_unique)
            for k, i in enumerate(uniq):
                _link(files[i], cin / frame_name(k + 1))
            face_share = 0.3 if faces and faces.enabled else 0.0
            dup_note = f", {n_dup} repeated frames reused" if n_dup else ""

            def chunk_progress(frac: float, what: str):
                """frac = 0..1 through this chunk; the ETA counts every frame, skipped ones too."""
                if not ctl:
                    return
                frames_now = done_frames + frac * n_files
                speed_done = frames_now - rate_f0
                elapsed = time.time() - rate_t0
                eta = elapsed / speed_done * max(0, est_total - frames_now) if speed_done >= 1 else None
                msg = f"{chunk_label}: {what}" + (f", about {human_time(eta)} left" if eta is not None else "")
                ctl.progress(p0 + span * min(1.0, frames_now / est_total), msg, eta)

            # 3b. AI upscale of the unique frames
            cur = cin
            for pi, s in enumerate(plan.passes):
                dst = cout if pi == len(plan.passes) - 1 else work_root / f"chunk_pass{pi}"
                shutil.rmtree(dst, ignore_errors=True)
                dst.mkdir()
                name = engines.model_name_for_scale(model_name, s)

                def on_up(pi=pi, dst=dst):
                    # Runs once a second from run_tool, whether or not the engine prints anything.
                    if not ctl:
                        return
                    made = min(_count_frames(dst), n_unique)
                    frac = (pi + made / n_unique) / len(plan.passes) * (1 - face_share)
                    if made == 0 and done_frames + frac * n_files - rate_f0 < 1:
                        elapsed = time.time() - rate_t0
                        ctl.progress(p0 + span * min(1.0, done_frames / est_total),
                                     f"{chunk_label}: AI engine warming up "
                                     f"({human_time(elapsed)} so far, the first frame is the slowest)")
                        return
                    chunk_progress(frac, f"AI upscaling {made}/{n_unique} frames" + dup_note
                                   + (f" (pass {pi + 1}/{len(plan.passes)})" if len(plan.passes) > 1 else ""))

                tile = ctx.tile(s)
                stall = STALL_SECONDS_CPU if ctx.cpu else STALL_SECONDS_GPU
                for attempt in range(3):
                    try:
                        run_tool(engines.realesrgan_cmd(cur, dst, name, s, tile, ctx.engine_gpu, ctx.threads, "png"),
                                 ctl, name="Real-ESRGAN", on_tick=on_up,
                                 progress_counter=lambda dst=dst: _count_frames(dst), stall_seconds=stall)
                        break
                    except EngineStalled:
                        smaller = max(MIN_RETRY_TILE, (tile or 256) // 2)
                        if attempt == 2 or smaller == tile:
                            raise RuntimeError(
                                "The AI engine stopped responding on your graphics card, so the job was paused "
                                "instead of waiting forever. Update your graphics driver, or in Settings pick a "
                                "smaller Tile size or a different GPU, then press Retry. "
                                "Finished chunks are kept.")
                        log.warning("Real-ESRGAN stalled with tile %s; retrying chunk %d with tile %s",
                                    tile, ci + 1, smaller)
                        tile = smaller
                        if ctl:
                            ctl.progress(p0 + span * min(1.0, done_frames / est_total),
                                         f"{chunk_label}: engine got stuck, retrying with a smaller tile ({tile})")
                if cur not in (cin,):
                    shutil.rmtree(cur, ignore_errors=True)
                cur = dst
            if _count_frames(cur) < n_unique:
                raise RuntimeError("The AI engine skipped some frames. Try a smaller tile size in Settings.")

            # 3c. optional face enhancement (unique frames only)
            if faces and faces.enabled:
                faces.run(cur, n_unique, ctl,
                          lambda k: chunk_progress((1 - face_share) + face_share * k / n_unique,
                                                   f"enhancing faces {k}/{n_unique}"))

            # 3d. put the repeated frames back so the chunk has every frame, in order
            if n_dup:
                for i, pos in enumerate(dd.output_positions(ref)):
                    _link(cur / frame_name(pos + 1), cfull / frame_name(i + 1))
                enc_dir = cfull
            else:
                enc_dir = cur
            if _count_frames(enc_dir) != n_files:
                raise RuntimeError("Some frames went missing before encoding. Press Retry to redo this part.")
            if ctl:
                ctl.check()

            def encode(enc):
                run_tool(fu.segment_encode_cmd(ffmpeg, str(enc_dir / PNG_PATTERN), fps, str(seg), enc, scale_vf,
                                               opts.quality, opts.crf, opts.bitrate_kbps, height=geo.canvas_h),
                         ctl, name="FFmpeg encode")
            try:
                encode(encoder)
            except RuntimeError:
                # A hardware encoder that passed the small start-up test can still refuse a big
                # frame size (6K/8K) or run out of video memory. Switch to the CPU encoder, but only
                # before any segment exists, so every segment of one video uses the same encoder.
                sw = fu.software_encoder(codec, fu.working_encoders(ffmpeg))
                if done or encoder == sw:
                    raise
                log.warning("Encoder %s failed at %dx%d; switching to %s", encoder, geo.canvas_w, geo.canvas_h, sw)
                if ctl:
                    ctl.progress(p0 + span * min(1.0, done_frames / est_total),
                                 f"{chunk_label}: hardware encoder refused this size, using the CPU encoder")
                encoder = sw
                state["encoder"] = encoder
                encode(encoder)
            skipped[str(ci)] = n_dup
            state.data["skipped"] = skipped
            done.add(ci)
            state["done_chunks"] = sorted(done)
            done_frames += n_files
            for d in (cin, cout, cfull):
                shutil.rmtree(d, ignore_errors=True)
            # Bounded temp space: this chunk's source frames are not needed any more.
            delete_frames(a, end)
            source.release(end)
            elapsed = time.time() - rate_t0
            per_frame = elapsed / max(1, done_frames - rate_f0)
            eta = per_frame * max(0, est_total - done_frames)
            n_skipped = sum(skipped.values())
            prog(p0 + span * min(1.0, done_frames / est_total),
                 f"Upscaled {done_frames}/{est_total} frames"
                 + (f" ({n_skipped} repeated frames skipped)" if n_skipped else "")
                 + f" ({human_time(eta)} left)", eta)
            if end < b:
                break   # last (short) chunk
            ci += 1
    finally:
        source.stop()

    if not total:
        total = done_frames
    state["frames_total"] = total
    n_skipped = sum(skipped.values())
    if opts.skip_duplicates:
        log.info("Duplicate-frame skipping: %d of %d frames (%.1f%%) reused an upscaled frame instead of "
                 "running the AI again", n_skipped, total, 100.0 * n_skipped / max(1, total))
    if faces:
        log.info("Face enhancement: %s", f"{faces.faces_found} face(s) restored" if faces.ran else "skipped")

    # -------- 4. join segments + original audio/subtitles
    prog(0.95, "Adding the original audio and subtitles")
    segs = [str(seg_path(i)) for i in range(n_chunks)]
    missing = [s_ for s_ in segs if not Path(s_).exists()]
    if missing or not segs:
        raise RuntimeError("Some finished parts of this video are missing. Discard the job and start again.")
    lst = work_root / "segments.txt"
    lst.write_text(fu.concat_list_text(segs), encoding="utf-8")
    if out_path is None:
        tag = opts.preset.split(" ")[0] + ("_60fps" if opts.interpolate else "") + "_enhanced"
        out = unique_output_path(src, out_dir, ext, tag)
    else:
        out = Path(out_path)
    out = _mux_with_fallback(ffmpeg, lst, out, src, info, ext, start, duration, ctl)

    # -------- 5. clean up temporary frames
    if not keep_temp:
        shutil.rmtree(work_root, ignore_errors=True)
        shutil.rmtree(jd, ignore_errors=True)
    log.info("Video done in %s: %s", human_time(time.time() - t_start), out)
    if ctl:
        ctl.progress(1.0, f"Saved {out.name}")
    return str(out)


def _mux_with_fallback(ffmpeg: str, lst: Path, out: Path, src: str, info: fu.MediaInfo, ext: str,
                       start: Optional[float], duration: Optional[float], ctl: Optional[JobControl]) -> Path:
    """Join into the chosen container. MP4/MOV are written with 64-bit offsets as needed
    (files over 4 GB are fine) and +faststart; if the MP4/MOV mux still fails (an
    audio or subtitle stream MP4 cannot hold, a size limit), save as MKV instead of
    losing hours of work."""
    tmp_out = out.with_name(out.stem + ".partial" + ext)
    try:
        run_tool(fu.mux_cmd(ffmpeg, str(lst), str(tmp_out), src, info, ext, start, duration), ctl, name="FFmpeg mux")
        os.replace(tmp_out, out)
        return out
    except RuntimeError:
        if ext not in (".mp4", ".mov"):
            raise
        try:
            tmp_out.unlink()
        except OSError:
            pass
        mkv = out.with_suffix(".mkv")
        n = 2
        while mkv.exists():
            mkv = out.with_name(f"{out.stem}_{n}.mkv")
            n += 1
        log.warning("%s output failed; saving as MKV instead: %s", ext, mkv)
        if ctl:
            ctl.progress(0.96, f"{ext.upper().lstrip('.')} could not hold this video; saving as MKV instead")
        tmp_mkv = mkv.with_name(mkv.stem + ".partial.mkv")
        run_tool(fu.mux_cmd(ffmpeg, str(lst), str(tmp_mkv), src, info, ".mkv", start, duration), ctl,
                 name="FFmpeg mux (MKV)")
        os.replace(tmp_mkv, mkv)
        return mkv


def preview_clip(src: str, opts: VideoOptions, ctx: EngineContext, ctl: Optional[JobControl] = None,
                 at: Optional[float] = None, seconds: float = 5.0) -> str:
    """Process a 5-second sample (from the middle by default) to a temp file."""
    info = fu.probe(str(engines.ffprobe_path()), src)
    if at is None:
        at = max(0.0, info.duration / 2 - seconds / 2) if info.duration > seconds else 0.0
    seconds = min(seconds, max(0.5, info.duration - at)) if info.duration else seconds
    out = ctx.temp_dir / f"preview_{Path(src).stem}_{int(time.time())}{VIDEO_CONTAINERS[opts.container]}"
    return enhance_video(src, opts, ctx, ctl, start=at, duration=seconds, out_path=str(out))


def unfinished_jobs() -> List[dict]:
    """Jobs that can be resumed (shown on the Home tab at start-up)."""
    res = []
    if not JOBS_DIR.exists():
        return res
    for st in JOBS_DIR.glob("*/state.json"):
        try:
            data = json.loads(st.read_text(encoding="utf-8"))
            if (data.get("stage") != "new" and not data.get("preview")
                    and Path(data.get("source", "")).exists()):
                data["id"] = st.parent.name
                res.append(data)
        except Exception:
            continue
    return res


def discard_job(job_id: str):
    st = JOBS_DIR / job_id / "state.json"
    try:
        data = json.loads(st.read_text(encoding="utf-8"))
        shutil.rmtree(data.get("work", ""), ignore_errors=True)
    except Exception:
        pass
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
