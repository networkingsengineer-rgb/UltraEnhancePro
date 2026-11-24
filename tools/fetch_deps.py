"""Download FFmpeg, Real-ESRGAN, RIFE and the face models into ./vendor.

Run once by build.bat on the build machine. The end user never runs this:
everything ends up inside the installer and the portable ZIP.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"

DEPS = {
    "realesrgan": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
                  "realesrgan-ncnn-vulkan-20220424-windows.zip",
    "rife": "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/"
            "rife-ncnn-vulkan-20221029-windows.zip",
    "ffmpeg": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
              "ffmpeg-master-latest-win64-gpl.zip",
}
MODELS = {
    "gfpgan_1.4.onnx": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/"
                                         "face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def download(url: str) -> bytes:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "UltraEnhancePro-build"})
    with urllib.request.urlopen(req, timeout=600) as r:
        total = int(r.headers.get("Content-Length") or 0)
        buf = io.BytesIO()
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total:3d}%  {done >> 20} MB", end="", flush=True)
        print()
        return buf.getvalue()


def unzip_flat(data: bytes, dest: Path, keep=None):
    """Extract a zip, dropping its single top-level folder."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        prefix = ""
        tops = {n.split("/")[0] for n in names}
        if len(tops) == 1 and all("/" in n for n in names):
            prefix = next(iter(tops)) + "/"
        for n in names:
            rel = n[len(prefix):]
            if keep and not keep(rel):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)


def main(force: bool = False) -> int:
    VENDOR.mkdir(exist_ok=True)
    checks = {
        "realesrgan": VENDOR / "realesrgan" / "realesrgan-ncnn-vulkan.exe",
        "rife": VENDOR / "rife" / "rife-ncnn-vulkan.exe",
        "ffmpeg": VENDOR / "ffmpeg" / "ffmpeg.exe",
    }
    for name, url in DEPS.items():
        if checks[name].exists() and not force:
            print(f"[ok] {name} already present")
            continue
        print(f"[get] {name}")
        data = download(url)
        if name == "ffmpeg":
            tmp = VENDOR / "_ffmpeg"
            shutil.rmtree(tmp, ignore_errors=True)
            unzip_flat(data, tmp)
            dst = VENDOR / "ffmpeg"
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir()
            for exe in ("ffmpeg.exe", "ffprobe.exe"):
                shutil.copy2(tmp / "bin" / exe, dst / exe)
            if (tmp / "LICENSE.txt").exists():
                shutil.copy2(tmp / "LICENSE.txt", dst / "LICENSE.txt")
            shutil.rmtree(tmp, ignore_errors=True)
        elif name == "realesrgan":
            # The zip has no top folder and includes sample images/videos we don't need.
            unzip_flat(data, VENDOR / "realesrgan",
                       keep=lambda r: not r.lower().endswith((".jpg", ".png", ".mp4", ".gif")))
        else:
            unzip_flat(data, VENDOR / "rife")
        if not checks[name].exists():
            print(f"[error] {name}: expected {checks[name]} after extraction")
            return 1
    mdir = VENDOR / "models"
    mdir.mkdir(exist_ok=True)
    for fname, url in MODELS.items():
        p = mdir / fname
        if p.exists() and p.stat().st_size > 100_000 and not force:
            print(f"[ok] {fname} already present")
            continue
        print(f"[get] {fname}")
        p.write_bytes(download(url))
        print(f"       sha1 {hashlib.sha1(p.read_bytes()).hexdigest()}")
    print("\nAll engines and models are in", VENDOR)
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))
