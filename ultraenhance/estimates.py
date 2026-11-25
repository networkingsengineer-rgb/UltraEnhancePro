"""Honest disk-space and time estimates, plus free-space checks."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

# Average PNG size per pixel for natural video frames (RGB, compressed).
PNG_BYTES_PER_PIXEL = 1.6

# Rough Real-ESRGAN throughput in output megapixels per second for a
# mid-range GPU (RTX 3060 / RX 6600 class). CPU is ~40x slower.
GPU_MPIX_PER_SEC = 2.2
CPU_MPIX_PER_SEC = 0.05


def free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def human_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m} min"
    if m:
        return f"{m} min {s} s"
    return f"{s} s"


def video_temp_bytes(src_w: int, src_h: int, ai_w: int, ai_h: int, frames: int,
                     interpolate_factor: float = 1.0) -> int:
    """Temp space for extracted + upscaled frames, processed in chunks.

    Input frames are kept for the whole job (needed for resume); upscaled
    frames are encoded per chunk but when interpolating they are kept until
    RIFE finishes, so we count all of them.
    """
    inp = src_w * src_h * PNG_BYTES_PER_PIXEL * frames
    out = ai_w * ai_h * PNG_BYTES_PER_PIXEL * frames
    if interpolate_factor > 1:
        out *= (1 + interpolate_factor)
    return int((inp + out) * 1.1)


def video_output_bytes(out_w: int, out_h: int, duration: float, codec: str = "H.264",
                       quality: str = "Balanced", fps: float = 30.0,
                       bitrate_kbps: Optional[int] = None) -> int:
    """Approximate final file size: the target bitrate when one is set, otherwise
    ~80 % of the recommended bitrate for this size/codec/quality (CRF files
    usually come in a little under it)."""
    from .ffmpeg_utils import recommended_bitrate_kbps
    if bitrate_kbps:
        kbps = bitrate_kbps
    else:
        kbps = 0.8 * recommended_bitrate_kbps(out_w, out_h, fps, codec, quality)
    return int(kbps * 1000 / 8 * max(0.0, duration))


def video_temp_bytes_streamed(src_w: int, src_h: int, ai_w: int, ai_h: int, chunk: int,
                              frames_ahead: int, output_bytes: int, passes: int = 1) -> int:
    """Peak temporary space when frames are decoded a few chunks ahead and deleted
    after each chunk is encoded: source frames in flight + one chunk of upscaled
    frames (two when there is an intermediate AI pass) + the encoded segments
    (which add up to roughly the final file) + 10 % slack."""
    src = src_w * src_h * PNG_BYTES_PER_PIXEL * (chunk + frames_ahead)
    up = ai_w * ai_h * PNG_BYTES_PER_PIXEL * chunk * (2 if passes > 1 else 1)
    return int((src + up + output_bytes) * 1.1)


# Relative speed of the AI models against realesrgan-x4plus (the rates above).
# realesr-animevideov3 is a small SRVGG network, several times faster.
MODEL_SPEED = {"realesr-animevideov3": 6.0}


def video_seconds(ai_w: int, ai_h: int, frames: int, passes: int = 1, cpu: bool = False,
                  interpolate_factor: float = 1.0, model: str = "") -> float:
    rate = (CPU_MPIX_PER_SEC if cpu else GPU_MPIX_PER_SEC) * MODEL_SPEED.get(model, 1.0)
    mpix = ai_w * ai_h / 1e6
    t = frames * mpix * max(1, passes) / rate
    if interpolate_factor > 1:
        t *= 1.35
    return t + frames * 0.02  # decode/encode overhead


def photo_seconds(ai_w: int, ai_h: int, passes: int = 1, cpu: bool = False) -> float:
    rate = CPU_MPIX_PER_SEC if cpu else GPU_MPIX_PER_SEC
    return 1.5 + (ai_w * ai_h / 1e6) * max(1, passes) / rate


def check_space(path: str | Path, needed: int) -> tuple[bool, str]:
    free = free_bytes(path)
    if free < needed:
        return False, (f"Not enough free disk space on {Path(path).anchor or path}. "
                       f"Needed about {human_bytes(needed)}, free {human_bytes(free)}. "
                       f"Free up space or choose another temp/output folder in Settings.")
    return True, f"{human_bytes(free)} free, about {human_bytes(needed)} needed."
