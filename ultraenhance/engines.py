"""Locating and calling the bundled native tools (FFmpeg, Real-ESRGAN, RIFE)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import vendor_dir

EXE = ".exe" if os.name == "nt" else ""

# Keep console windows from flashing on Windows.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class EngineMissing(RuntimeError):
    pass


def _find(rel: str, fallback_name: str) -> Path:
    p = vendor_dir() / rel
    if p.exists():
        return p
    found = shutil.which(fallback_name)
    if found:
        return Path(found)
    raise EngineMissing(
        f"{fallback_name} was not found. The installation looks incomplete. "
        f"Please reinstall UltraEnhance Pro (expected at {p})."
    )


def ffmpeg_path() -> Path:
    return _find(f"ffmpeg/ffmpeg{EXE}", "ffmpeg")


def ffprobe_path() -> Path:
    return _find(f"ffmpeg/ffprobe{EXE}", "ffprobe")


def realesrgan_path() -> Path:
    return _find(f"realesrgan/realesrgan-ncnn-vulkan{EXE}", "realesrgan-ncnn-vulkan")


def realesrgan_models_dir() -> Path:
    return realesrgan_path().parent / "models"


def rife_path() -> Path:
    return _find(f"rife/rife-ncnn-vulkan{EXE}", "rife-ncnn-vulkan")


def rife_model_dir() -> Path:
    base = rife_path().parent
    for name in ("rife-v4.6", "rife-v4", "rife-anime", "rife-v2.4"):
        if (base / name).exists():
            return base / name
    return base / "rife-v4.6"


def face_models() -> tuple[Path, Path]:
    d = vendor_dir() / "models"
    return d / "gfpgan_1.4.onnx", d / "face_detection_yunet_2023mar.onnx"


def popen(cmd: List[str], **kw) -> subprocess.Popen:
    kw.setdefault("stdout", subprocess.PIPE)
    kw.setdefault("stderr", subprocess.STDOUT)
    kw.setdefault("text", True)
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    if os.name == "nt":
        kw.setdefault("creationflags", CREATE_NO_WINDOW)
    return subprocess.Popen([str(c) for c in cmd], **kw)


def run(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = CREATE_NO_WINDOW
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, **kw)


def realesrgan_cmd(inp: str | Path, out: str | Path, model: str, scale: int,
                   tile: int = 0, gpu_id: int = -2, threads: str = "1:2:2",
                   fmt: str = "png", tta: bool = False) -> List[str]:
    """Build a Real-ESRGAN ncnn-vulkan command. inp/out may be files or folders."""
    cmd = [str(realesrgan_path()), "-i", str(inp), "-o", str(out),
           "-n", model, "-s", str(scale), "-t", str(tile),
           "-m", str(realesrgan_models_dir()), "-j", threads, "-f", fmt]
    if gpu_id != -2:  # -2 = let the engine pick the default GPU
        cmd += ["-g", str(gpu_id)]
    if tta:
        cmd.append("-x")
    return cmd


def model_name_for_scale(model: str, scale: int) -> str:
    """realesr-animevideov3 ships as -x2/-x3/-x4 param files."""
    if model == "realesr-animevideov3":
        return f"realesr-animevideov3-x{scale}" if scale in (2, 3, 4) else "realesr-animevideov3-x4"
    return model


def rife_cmd(inp: str | Path, out: str | Path, target_frames: int,
             gpu_id: int = -2, threads: str = "1:2:2", uhd: bool = False) -> List[str]:
    cmd = [str(rife_path()), "-i", str(inp), "-o", str(out),
           "-n", str(target_frames), "-m", str(rife_model_dir()), "-j", threads,
           "-f", "%08d.png"]
    if gpu_id != -2:
        cmd += ["-g", str(gpu_id)]
    if uhd:
        cmd.append("-u")
    return cmd


def engines_status() -> dict:
    status = {}
    for name, fn in (("FFmpeg", ffmpeg_path), ("Real-ESRGAN", realesrgan_path), ("RIFE", rife_path)):
        try:
            status[name] = str(fn())
        except EngineMissing:
            status[name] = None
    g, y = face_models()
    status["Face restore (GFPGAN)"] = str(g) if g.exists() and y.exists() else None
    return status
