"""Resolution presets and the scaling maths used by the photo and video pipelines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Target heights for landscape media (short side for portrait).
RESOLUTION_PRESETS = {
    "720p HD": (1280, 720),
    "1080p Full HD": (1920, 1080),
    "2K (1440p)": (2560, 1440),
    "4K (2160p)": (3840, 2160),
    "8K (4320p)": (7680, 4320),
}
# Output sizes offered for video (the photo side keeps RESOLUTION_PRESETS above).
VIDEO_RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p Full HD": (1920, 1080),
    "2K (1440p QHD)": (2560, 1440),
    "4K (2160p UHD)": (3840, 2160),
    "6K (3240p)": (5760, 3240),
    "8K (4320p)": (7680, 4320),
}
# Older video preset names (saved jobs / settings) -> current name. Both keep working.
VIDEO_PRESET_ALIASES = {"720p HD": "720p", "2K (1440p)": "2K (1440p QHD)", "4K (2160p)": "4K (2160p UHD)"}


def canonical_video_preset(name: str, default: str = "1080p Full HD") -> str:
    """Map any known preset name (old or new) to the name shown in the video list."""
    name = VIDEO_PRESET_ALIASES.get(name, name)
    return name if name in VIDEO_RESOLUTION_PRESETS else default


SCALE_PRESETS = {"2x": 2, "3x": 3, "4x": 4, "8x": 8}

FIT_MODES = ("fit", "fill", "pad")

# AI models -> (ncnn model name, native scales the model supports)
MODELS = {
    "Photo (realistic)": ("realesrgan-x4plus", (4,)),
    "Anime / Art": ("realesrgan-x4plus-anime", (4,)),
    "Old photo restore": ("realesrgan-x4plus", (4,)),
    "Video (anime/cartoon, fast)": ("realesr-animevideov3", (2, 3, 4)),
    # Video defaults: the small video model is roughly 10x faster than realesrgan-x4plus
    # and runs at the exact 2x/3x/4x needed instead of always computing 4x.
    "Fast (recommended)": ("realesr-animevideov3", (2, 3, 4)),
    "Best quality (slow)": ("realesrgan-x4plus", (4,)),
}

MAX_QUALITY = "Max quality (large file)"
QUALITY_PRESETS = {
    # name: (x264/x265 preset, crf, hw preset)
    "Fast": ("veryfast", 23, "p2"),
    "Balanced": ("medium", 20, "p5"),
    "Best": ("slow", 17, "p7"),
    # Near-transparent: low CRF, big files. Meant for archiving or further editing.
    MAX_QUALITY: ("slow", 14, "p7"),
}


# One-click video modes: name -> the VideoOptions fields the mode sets. Every mode
# sets the same fields so switching modes never leaves one mode's extras behind.
# Fields not listed (resolution, codec, container, ...) are left as they are.
VIDEO_MODE_CUSTOM = "Custom"
VIDEO_MODES = {
    "Fast HD": {
        "model": "Fast (recommended)", "skip_duplicates": True, "face_enhance": False,
        "cleanup": "off", "auto_color": False, "sharpen": False,
        "deinterlace": "auto", "denoise": False, "quality": "Balanced",
    },
    "Restore old video": {
        "model": "Fast (recommended)", "skip_duplicates": True, "face_enhance": True,
        "cleanup": "strong", "auto_color": True, "sharpen": True,
        "deinterlace": "auto", "denoise": True,
    },
    "Best quality (slow)": {
        "model": "Best quality (slow)", "skip_duplicates": True, "face_enhance": True,
        "cleanup": "light", "auto_color": False, "sharpen": False,
        "deinterlace": "auto", "denoise": False, "quality": "Best",
    },
}


def apply_video_mode(values: dict, mode: str) -> dict:
    """Return a copy of an options dict with a mode's settings applied (Custom = unchanged)."""
    out = dict(values)
    out.update(VIDEO_MODES.get(mode, {}))
    return out


def detect_video_mode(values: dict) -> str:
    """The mode whose settings all match these options, else Custom."""
    for name, overrides in VIDEO_MODES.items():
        if all(values.get(k) == v for k, v in overrides.items()):
            return name
    return VIDEO_MODE_CUSTOM


@dataclass(frozen=True)
class Geometry:
    """Final placement of the upscaled image on the output canvas."""
    canvas_w: int
    canvas_h: int
    scaled_w: int
    scaled_h: int
    offset_x: int  # >=0 for pad, <=0 for fill (crop)
    offset_y: int


def even(n: float) -> int:
    """Round to the nearest even integer >= 2 (video encoders need even sizes)."""
    v = int(round(n / 2.0)) * 2
    return max(2, v)


# Every preset name accepted by target_box: photo names, video names and old video names.
_RES_LOOKUP = dict(RESOLUTION_PRESETS)
_RES_LOOKUP.update(VIDEO_RESOLUTION_PRESETS)


def target_box(preset: str, src_w: int, src_h: int) -> Tuple[int, int]:
    """Return the (w, h) box for a resolution preset, rotated for portrait sources."""
    if preset not in _RES_LOOKUP:
        raise KeyError(f"Unknown preset: {preset}")
    w, h = _RES_LOOKUP[preset]
    if src_h > src_w:  # portrait: swap the box
        w, h = h, w
    return w, h


def fit_size(src_w: int, src_h: int, box_w: int, box_h: int) -> Tuple[int, int]:
    """Largest size with the source aspect ratio that fits inside the box."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source size must be positive")
    s = min(box_w / src_w, box_h / src_h)
    return max(1, round(src_w * s)), max(1, round(src_h * s))


def fill_size(src_w: int, src_h: int, box_w: int, box_h: int) -> Tuple[int, int]:
    """Smallest size with the source aspect ratio that covers the whole box."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source size must be positive")
    s = max(box_w / src_w, box_h / src_h)
    return max(box_w, round(src_w * s)), max(box_h, round(src_h * s))


def compute_geometry(src_w: int, src_h: int, preset: Optional[str] = None,
                     scale: Optional[float] = None, mode: str = "fit",
                     even_dims: bool = False) -> Geometry:
    """Work out output size.

    * scale given: plain multiply, aspect kept, no canvas.
    * preset + fit: image fits inside the box, canvas == image (no bars).
    * preset + fill: image covers the box and is centre-cropped to it.
    * preset + pad: image fits inside and is letter/pillar-boxed to the box.
    """
    if mode not in FIT_MODES:
        raise ValueError(f"mode must be one of {FIT_MODES}")
    rnd = even if even_dims else (lambda v: max(1, int(round(v))))
    if scale is not None:
        w, h = rnd(src_w * scale), rnd(src_h * scale)
        return Geometry(w, h, w, h, 0, 0)
    if preset is None:
        raise ValueError("Give either preset or scale")
    bw, bh = target_box(preset, src_w, src_h)
    if mode == "fit":
        w, h = fit_size(src_w, src_h, bw, bh)
        w, h = rnd(w), rnd(h)
        return Geometry(w, h, w, h, 0, 0)
    if mode == "fill":
        w, h = fill_size(src_w, src_h, bw, bh)
        w, h = rnd(w), rnd(h)
        bw, bh = rnd(bw), rnd(bh)
        return Geometry(bw, bh, w, h, -((w - bw) // 2), -((h - bh) // 2))
    w, h = fit_size(src_w, src_h, bw, bh)
    w, h = rnd(w), rnd(h)
    bw, bh = rnd(bw), rnd(bh)
    return Geometry(bw, bh, w, h, (bw - w) // 2, (bh - h) // 2)


def required_factor(src_w: int, src_h: int, geo: Geometry) -> float:
    """How much bigger the AI output must be (before final resize)."""
    return max(geo.scaled_w / src_w, geo.scaled_h / src_h)


def plan_passes(factor: float, native_scales: Tuple[int, ...] = (4,),
                max_passes: int = 3) -> List[int]:
    """Choose AI passes whose product is >= factor (minimal overshoot).

    Returns [] if no AI upscale is needed (factor <= 1). 8x with a 4x model
    becomes [4, 4]; 3x with an animevideov3 model becomes [3].
    """
    if factor <= 1.0 + 1e-6:
        return []
    best: Optional[List[int]] = None
    best_key = None

    def search(prefix: List[int], prod: int):
        nonlocal best, best_key
        if prod >= factor - 1e-6:
            key = (prod, len(prefix))
            if best_key is None or key < best_key:
                best, best_key = list(prefix), key
            return
        if len(prefix) >= max_passes:
            return
        for s in sorted(native_scales):
            search(prefix + [s], prod * s)

    search([], 1)
    if best is None:  # factor too large: use as many max passes as allowed
        best = [max(native_scales)] * max_passes
    return best


def describe_output(src_w: int, src_h: int, geo: Geometry) -> str:
    return f"{src_w}x{src_h} -> {geo.canvas_w}x{geo.canvas_h}"
