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
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

from . import engines, ffmpeg_utils as fu
from .config import JOBS_DIR, log
from .estimates import check_space, human_bytes, human_time, video_output_bytes, video_seconds
from .formats import VIDEO_CONTAINERS
from .job_queue import JobControl
from .photo import EngineContext, unique_output_path
from .presets import MODELS, compute_geometry, plan_passes, required_factor
from .runner import EngineStalled, percent_from_line, run_tool

PNG_PATTERN = "%08d.png"


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

    def summary(self) -> str:
        g = self.geo
        lines = [
            f"Source: {self.info.width}x{self.info.height}, {float(self.info.fps):.3f} fps, "
            f"{human_time(self.info.duration)}, {self.frames_in} frames",
            f"Output: {g.canvas_w}x{g.canvas_h} at {float(self.out_fps):.3f} fps ({self.frames_out} frames)",
            f"AI passes: {' + '.join(f'{p}x' for p in self.passes) or 'none (resize only)'}",
            f"Temporary disk space needed: about {human_bytes(self.temp_bytes)}",
            f"Estimated output file size: about {human_bytes(self.output_bytes)}",
            f"Estimated time: about {human_time(self.seconds)} (real time depends on your GPU)",
        ]
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
    png = 1.6
    temp = (info.width * info.height * png * (frames_in + frames_out)
            + ai_w * ai_h * png * chunk * 2
            + video_output_bytes(geo.canvas_w, geo.canvas_h, dur, opts.codec) * 1.2)
    out_bytes = video_output_bytes(geo.canvas_w, geo.canvas_h, dur, opts.codec)
    secs = video_seconds(ai_w, ai_h, frames_out, 1, ctx.cpu, 1.0) + (frames_in * 0.03 if opts.interpolate else 0)
    warnings = []
    if geo.canvas_h >= 4320 or geo.canvas_w >= 7680:
        warnings.append("8K video is extremely slow and the file will be very large. "
                        "Try the 5-second preview first.")
    elif geo.canvas_h >= 2160 or geo.canvas_w >= 3840:
        warnings.append("4K video takes a long time. A 5-second preview is recommended first.")
    if ctx.cpu:
        warnings.append("No GPU selected: CPU mode is many times slower.")
    if opts.container == "WEBM" and opts.codec != "AV1":
        warnings.append("WEBM only supports AV1 here, so AV1 will be used.")
    return VideoPlan(info, geo, passes, ai_w, ai_h, out_fps, frames_in, frames_out,
                     int(temp), int(out_bytes), secs, warnings)


def chunk_size(opts: VideoOptions, ai_w: int, ai_h: int) -> int:
    if opts.chunk_frames:
        return opts.chunk_frames
    mpix = ai_w * ai_h / 1e6
    # ~1.5 GB of upscaled PNGs per chunk at most, 24..240 frames.
    return int(max(24, min(240, 1500 / max(1.0, mpix * 1.6))))


def job_dir_for(src: str, opts: VideoOptions, start: Optional[float], duration: Optional[float]) -> Path:
    st = os.stat(src)
    key = json.dumps([os.path.abspath(src).lower(), st.st_size, int(st.st_mtime),
                      asdict(opts) | {"output_dir": ""}, start, duration], sort_keys=True, default=str)
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


def _encoder_for(opts: VideoOptions) -> str:
    codec = "AV1" if opts.container == "WEBM" else opts.codec
    available = fu.working_encoders(str(engines.ffmpeg_path()))
    enc = fu.choose_encoder(codec, available, opts.encoder_family)
    log.info("Encoder chosen: %s (codec %s)", enc, codec)
    return enc


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

    if state["stage"] == "new":
        ok, msg = check_space(ctx.temp_dir, plan.temp_bytes)
        if not ok:
            raise RuntimeError(msg)
    out_dir = Path(opts.output_dir) if opts.output_dir else Path(src).parent / "UltraEnhance"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, msg = check_space(out_dir, plan.output_bytes * 2)
    if not ok:
        raise RuntimeError(msg)

    t_start = time.time()

    def prog(p, m, eta=None):
        if ctl:
            ctl.check()
            ctl.progress(p, m, eta)

    # -------- 1. decode (with optional stabilise / deinterlace / denoise)
    if state["stage"] in ("new", "extracting"):
        state["stage"] = "extracting"
        for f in in_dir.glob("*.png"):
            f.unlink()
        trf = None
        if opts.stabilize:
            prog(0.01, "Analysing camera shake (stabilise pass 1/2)")
            trf = str(work_root / "transforms.trf")
            run_tool(fu.vidstab_detect_cmd(ffmpeg, src, trf, start, duration), ctl, name="FFmpeg vidstab")
        deint = opts.deinterlace == "on" or (opts.deinterlace == "auto" and info.interlaced)
        filters = fu.pre_filter_chain(deint, opts.denoise, trf)
        total = plan.frames_in

        def on_dec(line):
            n = fu.parse_progress_frame(line)
            if n is not None and ctl:
                ctl.progress(0.02 + 0.08 * min(1, n / total), f"Extracting frames {n}/{total}")
        run_tool(fu.extract_frames_cmd(ffmpeg, src, str(in_dir / PNG_PATTERN), info.fps, filters,
                                       start, duration), ctl, on_dec, name="FFmpeg decode")
        n = _count_frames(in_dir)
        if n == 0:
            raise RuntimeError("No frames could be read from this video.")
        state.data["frames"] = n
        state["stage"] = "interpolating" if opts.interpolate else "upscaling"

    frames_dir, fps = in_dir, info.fps

    # -------- 2. frame interpolation (RIFE) on the source-resolution frames
    if opts.interpolate and plan.out_fps != info.fps:
        target_frames = int(round(state["frames"] * float(plan.out_fps) / info.fps_float))
        if state["stage"] == "interpolating" or _count_frames(interp_dir) < target_frames:
            state["stage"] = "interpolating"
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
            if _count_frames(interp_dir) == 0:
                raise RuntimeError("Frame interpolation produced no frames.")
            state["stage"] = "upscaling"
        frames_dir, fps = interp_dir, plan.out_fps

    # -------- 3. AI upscale + encode, chunk by chunk (resumable)
    frames = sorted(frames_dir.glob("*.png"))
    total = len(frames)
    if total == 0:
        raise RuntimeError("No frames were found to upscale.")
    chunk = chunk_size(opts, plan.ai_w, plan.ai_h)
    chunks = [frames[i:i + chunk] for i in range(0, total, chunk)]
    encoder = _encoder_for(opts)
    scale_vf = fu.scale_filter(geo.canvas_w, geo.canvas_h, geo.scaled_w, geo.scaled_h, geo.offset_x, geo.offset_y)
    model_name, _ = MODELS[opts.model]
    done = set(state["done_chunks"])
    done_frames = sum(len(chunks[i]) for i in done if i < len(chunks))
    rate_t0, rate_f0 = time.time(), done_frames
    if done:
        log.info("Resuming: %d/%d chunks already finished", len(done), len(chunks))

    for ci, files in enumerate(chunks):
        seg = seg_dir / f"seg_{ci:05d}.mkv"
        if ci in done and seg.exists():
            continue
        prog(0.18 + 0.75 * done_frames / total, f"Chunk {ci + 1}/{len(chunks)}")
        cin, cout = work_root / "chunk_in", work_root / "chunk_out"
        for d in (cin, cout):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir()
        for k, f in enumerate(files):
            _link(f, cin / f"{k + 1:08d}.png")
        cur = cin
        for pi, s in enumerate(plan.passes):
            dst = cout if pi == len(plan.passes) - 1 else work_root / f"chunk_pass{pi}"
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir()
            name = engines.model_name_for_scale(model_name, s)
            base = done_frames

            def on_up(base=base, pi=pi, dst=dst, n_files=len(files)):
                # Runs once a second from run_tool, whether or not the engine prints anything.
                if not ctl:
                    return
                made = min(_count_frames(dst), n_files)
                frac = (pi + made / n_files) / len(plan.passes)
                frames_now = base + frac * n_files
                speed_done = frames_now - rate_f0
                elapsed = time.time() - rate_t0
                eta = elapsed / speed_done * (total - frames_now) if speed_done >= 1 else None
                if made == 0 and frames_now - rate_f0 < 1:
                    msg = (f"Chunk {ci + 1}/{len(chunks)}: AI engine warming up "
                           f"({human_time(elapsed)} so far, the first frame is the slowest)")
                else:
                    msg = (f"Chunk {ci + 1}/{len(chunks)}: AI upscaling {made}/{n_files} frames"
                           + (f" (pass {pi + 1}/{len(plan.passes)})" if len(plan.passes) > 1 else "")
                           + (f", about {human_time(eta)} left" if eta is not None else ""))
                ctl.progress(0.18 + 0.75 * frames_now / total, msg, eta)

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
                        ctl.progress(0.18 + 0.75 * done_frames / total,
                                     f"Chunk {ci + 1}/{len(chunks)}: engine got stuck, retrying with a smaller tile ({tile})")
            if cur not in (cin,):
                shutil.rmtree(cur, ignore_errors=True)
            cur = dst
        if not plan.passes:
            cur = cin
        if _count_frames(cur) < len(files):
            raise RuntimeError("The AI engine skipped some frames. Try a smaller tile size in Settings.")
        if ctl:
            ctl.check()
        run_tool(fu.segment_encode_cmd(ffmpeg, str(cur / PNG_PATTERN), fps, str(seg), encoder, scale_vf,
                                       opts.quality, opts.crf, opts.bitrate_kbps), ctl, name="FFmpeg encode")
        done.add(ci)
        state["done_chunks"] = sorted(done)
        done_frames += len(files)
        shutil.rmtree(cin, ignore_errors=True)
        shutil.rmtree(cout, ignore_errors=True)
        elapsed = time.time() - rate_t0
        per_frame = elapsed / max(1, done_frames - rate_f0)
        eta = per_frame * (total - done_frames)
        prog(0.18 + 0.75 * done_frames / total,
             f"Upscaled {done_frames}/{total} frames ({human_time(eta)} left)", eta)

    # -------- 4. join segments + original audio/subtitles
    prog(0.95, "Adding the original audio and subtitles")
    segs = [str(seg_dir / f"seg_{i:05d}.mkv") for i in range(len(chunks))]
    lst = work_root / "segments.txt"
    lst.write_text(fu.concat_list_text(segs), encoding="utf-8")
    ext = VIDEO_CONTAINERS[opts.container]
    if out_path is None:
        tag = opts.preset.split(" ")[0] + ("_60fps" if opts.interpolate else "") + "_enhanced"
        out = unique_output_path(src, out_dir, ext, tag)
    else:
        out = Path(out_path)
    tmp_out = out.with_name(out.stem + ".partial" + ext)
    run_tool(fu.mux_cmd(ffmpeg, str(lst), str(tmp_out), src, info, ext, start, duration), ctl, name="FFmpeg mux")
    os.replace(tmp_out, out)

    # -------- 5. clean up temporary frames
    if not keep_temp:
        shutil.rmtree(work_root, ignore_errors=True)
        shutil.rmtree(jd, ignore_errors=True)
    log.info("Video done in %s: %s", human_time(time.time() - t_start), out)
    if ctl:
        ctl.progress(1.0, f"Saved {out.name}")
    return str(out)


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
