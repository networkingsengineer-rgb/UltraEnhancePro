"""Photo enhancement pipeline."""
from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import engines, face_restore
from .config import Settings, log
from .estimates import check_space, photo_seconds
from .job_queue import JobControl
from .presets import MODELS, compute_geometry, plan_passes, required_factor
from .runner import percent_from_line, run_tool
from .tiling import safe_tile_size

try:  # HEIC/HEIF support
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = None  # 8K outputs are legitimately huge
MAX_AI_SIDE = 16384


@dataclass
class PhotoOptions:
    preset: Optional[str] = "4K (2160p)"   # or None when using scale
    scale: Optional[int] = None             # 2/3/4/8
    fit_mode: str = "fit"
    model: str = "Photo (realistic)"
    face_restore: bool = False
    face_strength: float = 0.8
    denoise: bool = False
    sharpen: bool = False
    auto_fix: bool = False
    remove_jpeg_artifacts: bool = False
    out_format: str = "JPG"
    jpg_quality: int = 95
    keep_exif: bool = True
    output_dir: str = ""
    tta: bool = False

    def effective(self) -> "PhotoOptions":
        """'Old photo restore' switches on the clean-up tools automatically."""
        if self.model == "Old photo restore":
            o = PhotoOptions(**self.__dict__)
            o.face_restore = True
            o.denoise = True
            o.auto_fix = True
            o.remove_jpeg_artifacts = True
            return o
        return self


@dataclass
class EngineContext:
    gpu_id: int = -2
    vram_mb: int = 0
    cpu: bool = False
    tile_override: int = 0
    threads: str = "1:2:2"
    temp_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))

    def tile(self, scale: int) -> int:
        return self.tile_override or safe_tile_size(self.vram_mb, scale, self.cpu)

    @property
    def engine_gpu(self) -> int:
        return -1 if self.cpu else self.gpu_id


# ---------------------------------------------------------------- loading
def load_image(path: str) -> tuple[Image.Image, Optional[bytes], Optional[bytes]]:
    """Open any supported photo; returns (RGB/RGBA image, exif bytes, icc profile)."""
    try:
        img = Image.open(path)
        img.load()
    except Exception:
        # Fallback: let FFmpeg decode it (rare formats / broken headers)
        tmp = Path(tempfile.gettempdir()) / f"ue_{uuid.uuid4().hex}.png"
        from .ffmpeg_utils import photo_convert_cmd
        res = engines.run(photo_convert_cmd(str(engines.ffmpeg_path()), path, str(tmp)), timeout=120)
        if res.returncode != 0 or not tmp.exists():
            raise ValueError(f"This photo could not be opened: {Path(path).name}")
        img = Image.open(tmp)
        img.load()
        tmp.unlink(missing_ok=True)
    exif = img.info.get("exif")
    icc = img.info.get("icc_profile")
    if getattr(img, "is_animated", False):
        img.seek(0)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() or img.mode == "P" else "RGB")
    if exif:
        try:  # orientation is now baked in -> reset the tag so viewers don't rotate twice
            ex = Image.Exif()
            ex.load(exif)
            ex[0x0112] = 1
            exif = ex.tobytes()
        except Exception:
            pass
    return img, exif, icc


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def imwrite(path, arr) -> None:
    """cv2.imwrite that works with non-English (Unicode) Windows paths."""
    ok, buf = cv2.imencode(Path(path).suffix or ".png", arr)
    if not ok:
        raise RuntimeError(f"Could not encode image {path}")
    Path(path).write_bytes(buf.tobytes())


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


# ---------------------------------------------------------------- filters
def remove_jpeg_artifacts(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 5, 15)


def denoise(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)


def sharpen(bgr: np.ndarray, amount: float = 0.6, radius: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def auto_fix(bgr: np.ndarray) -> np.ndarray:
    """Gentle white balance + local contrast + brightness/contrast stretch."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(0) + 1e-6
    gray = means.mean()
    gain = np.clip(gray / means, 0.85, 1.15)       # gray-world, limited
    img = np.clip(img * (0.5 + 0.5 * gain), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    lo, hi = np.percentile(l, (0.5, 99.5))
    if hi - lo > 20:
        l = np.clip((l.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.06, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------- AI upscale
def ai_upscale(bgr: np.ndarray, model_key: str, passes: list, ctx: EngineContext,
               ctl: Optional[JobControl] = None, base_progress=(0.1, 0.8), tta: bool = False) -> np.ndarray:
    model, _ = MODELS[model_key]
    work = Path(tempfile.mkdtemp(prefix="ue_photo_", dir=ctx.temp_dir))
    try:
        cur = work / "in_0.png"
        imwrite(cur, bgr)
        lo, hi = base_progress
        for i, s in enumerate(passes):
            if ctl:
                ctl.check()
            out = work / f"out_{i}.png"
            name = engines.model_name_for_scale(model, s)
            cmd = engines.realesrgan_cmd(cur, out, name, s, ctx.tile(s), ctx.engine_gpu, ctx.threads, "png", tta)
            span = (hi - lo) / len(passes)

            def on_line(line, i=i, span=span):
                p = percent_from_line(line)
                if p is not None and ctl:
                    ctl.progress(lo + span * (i + p), f"AI upscaling pass {i + 1}/{len(passes)} ({s}x)")
            run_tool(cmd, ctl, on_line, name="Real-ESRGAN")
            if not out.exists():
                raise RuntimeError("The AI engine finished but produced no image. "
                                   "Try lowering the tile size in Settings.")
            cur = out
        res = imread(cur)
        if res is None:
            raise RuntimeError("Could not read the AI result image.")
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- output
def unique_output_path(src: str, out_dir: Path, suffix: str, tag: str) -> Path:
    stem = Path(src).stem
    p = out_dir / f"{stem}_{tag}{suffix}"
    n = 2
    src_res = Path(src).resolve()
    while p.exists() or p.resolve() == src_res:
        p = out_dir / f"{stem}_{tag}_{n}{suffix}"
        n += 1
    return p


def place_on_canvas(img: Image.Image, geo) -> Image.Image:
    img = img.resize((geo.scaled_w, geo.scaled_h), Image.LANCZOS) if img.size != (geo.scaled_w, geo.scaled_h) else img
    if (geo.canvas_w, geo.canvas_h) == (geo.scaled_w, geo.scaled_h):
        return img
    if geo.offset_x < 0 or geo.offset_y < 0:
        x, y = -min(0, geo.offset_x), -min(0, geo.offset_y)
        return img.crop((x, y, x + geo.canvas_w, y + geo.canvas_h))
    mode = img.mode
    canvas = Image.new(mode, (geo.canvas_w, geo.canvas_h), (0, 0, 0, 0) if mode == "RGBA" else (0, 0, 0))
    canvas.paste(img, (geo.offset_x, geo.offset_y))
    return canvas


def save_image(img: Image.Image, path: Path, fmt: str, quality: int, exif: Optional[bytes], icc: Optional[bytes]):
    kw = {}
    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    fmt = fmt.upper()
    if fmt == "JPG":
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        img.save(path, "JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2,
                 optimize=True, progressive=True, **kw)
    elif fmt == "PNG":
        img.save(path, "PNG", compress_level=6, **kw)
    elif fmt == "WEBP":
        img.save(path, "WEBP", quality=int(quality), method=5, **kw)
    elif fmt == "TIFF":
        img.save(path, "TIFF", compression="tiff_lzw", **kw)
    else:
        raise ValueError(f"Unknown output format {fmt}")


# ---------------------------------------------------------------- pipeline
def enhance_array(bgr: np.ndarray, alpha: Optional[np.ndarray], opts: PhotoOptions, ctx: EngineContext,
                  ctl: Optional[JobControl] = None) -> tuple[Image.Image, object]:
    """Run the whole pipeline on an image array; returns (final PIL image, geometry)."""
    o = opts.effective()
    h, w = bgr.shape[:2]
    geo = compute_geometry(w, h, o.preset if o.scale is None else None,
                           float(o.scale) if o.scale else None, o.fit_mode)
    factor = required_factor(w, h, geo)
    _, native = MODELS[o.model]
    passes = plan_passes(factor, native)

    # Keep AI output under a sane size by shrinking very large inputs first.
    prod = 1
    for s in passes:
        prod *= s
    if passes and max(w, h) * prod > MAX_AI_SIDE:
        shrink = MAX_AI_SIDE / (max(w, h) * prod)
        bgr = cv2.resize(bgr, (max(1, int(w * shrink)), max(1, int(h * shrink))), interpolation=cv2.INTER_AREA)

    def step(p, msg):
        if ctl:
            ctl.check()
            ctl.progress(p, msg)

    if o.remove_jpeg_artifacts:
        step(0.03, "Removing JPEG artefacts")
        bgr = remove_jpeg_artifacts(bgr)
    if o.denoise:
        step(0.06, "Reducing noise")
        bgr = denoise(bgr)
    if passes:
        step(0.1, "AI upscaling")
        bgr = ai_upscale(bgr, o.model, passes, ctx, ctl, (0.1, 0.8), o.tta)
    if o.face_restore:
        step(0.82, "Restoring faces")
        if face_restore.available():
            bgr, n = face_restore.restore_faces(bgr, o.face_strength, ctl.check if ctl else None)
            log.info("Face restore: %d face(s)", n)
        else:
            log.warning("Face restore models missing; skipped")
    if o.auto_fix:
        step(0.88, "Fixing colour, brightness and contrast")
        bgr = auto_fix(bgr)
    if o.sharpen:
        step(0.9, "Sharpening")
        bgr = sharpen(bgr)
    step(0.93, "Resizing to the final size")
    img = bgr_to_pil(bgr)
    if alpha is not None:
        a = Image.fromarray(alpha).resize(img.size, Image.LANCZOS)
        img.putalpha(a)
    return place_on_canvas(img, geo), geo


def enhance_photo(src: str, opts: PhotoOptions, ctx: EngineContext, ctl: Optional[JobControl] = None) -> str:
    t0 = time.time()
    img, exif, icc = load_image(src)
    alpha = np.asarray(img.split()[3]) if img.mode == "RGBA" else None
    bgr = pil_to_bgr(img)
    out_dir = Path(opts.output_dir) if opts.output_dir else Path(src).parent / "UltraEnhance"
    out_dir.mkdir(parents=True, exist_ok=True)

    o = opts.effective()
    geo = compute_geometry(img.width, img.height, o.preset if o.scale is None else None,
                           float(o.scale) if o.scale else None, o.fit_mode)
    needed = geo.canvas_w * geo.canvas_h * 3 * 4 + 50_000_000
    ok, msg = check_space(out_dir, needed)
    if not ok:
        raise RuntimeError(msg)
    ok, msg = check_space(ctx.temp_dir, needed * 3)
    if not ok:
        raise RuntimeError(msg)
    est = photo_seconds(geo.scaled_w, geo.scaled_h, cpu=ctx.cpu)
    if ctl:
        ctl.progress(0.01, "Loaded photo", eta=est)

    final, geo = enhance_array(bgr, alpha, opts, ctx, ctl)
    tag = (o.preset.split(" ")[0] if o.scale is None and o.preset else f"{o.scale}x") + "_enhanced"
    from .formats import PHOTO_OUTPUT_FORMATS
    out = unique_output_path(src, out_dir, PHOTO_OUTPUT_FORMATS[opts.out_format], tag)
    if ctl:
        ctl.progress(0.96, "Saving")
    save_image(final, out, opts.out_format, opts.jpg_quality, exif if opts.keep_exif else None, icc)
    log.info("Photo done: %s -> %s (%dx%d) in %.1fs", src, out, final.width, final.height, time.time() - t0)
    if ctl:
        ctl.progress(1.0, f"Saved {out.name} ({final.width}x{final.height})")
    return str(out)


def preview_crop(src: str, opts: PhotoOptions, ctx: EngineContext, crop_px: int = 320,
                 center: tuple[float, float] = (0.5, 0.5), ctl: Optional[JobControl] = None):
    """Enhance only a small region so the before/after preview is quick.

    Returns (before PIL, after PIL) at the same display size.
    """
    img, _, _ = load_image(src)
    w, h = img.size
    o = opts.effective()
    geo = compute_geometry(w, h, o.preset if o.scale is None else None,
                           float(o.scale) if o.scale else None, o.fit_mode)
    factor = max(1.0, required_factor(w, h, geo))
    cw, ch = min(crop_px, w), min(crop_px, h)
    cx, cy = int(center[0] * w - cw / 2), int(center[1] * h - ch / 2)
    cx, cy = max(0, min(w - cw, cx)), max(0, min(h - ch, cy))
    crop = img.crop((cx, cy, cx + cw, cy + ch)).convert("RGB")
    bgr = pil_to_bgr(crop)
    _, native = MODELS[o.model]
    passes = plan_passes(factor, native)
    if o.remove_jpeg_artifacts:
        bgr = remove_jpeg_artifacts(bgr)
    if o.denoise:
        bgr = denoise(bgr)
    if passes:
        bgr = ai_upscale(bgr, o.model, passes, ctx, ctl, (0.1, 0.9))
    if o.face_restore and face_restore.available():
        bgr, _ = face_restore.restore_faces(bgr, o.face_strength)
    if o.auto_fix:
        bgr = auto_fix(bgr)
    if o.sharpen:
        bgr = sharpen(bgr)
    out_size = (max(1, int(cw * factor)), max(1, int(ch * factor)))
    after = bgr_to_pil(bgr).resize(out_size, Image.LANCZOS)
    before = crop.resize(out_size, Image.BICUBIC)
    return before, after
