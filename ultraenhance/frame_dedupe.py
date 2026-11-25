"""Duplicate / near-static frame detection for video upscaling.

Many videos contain runs of identical or almost identical frames: static
title cards, slideshows, paused scenes, 24 fps film telecined to 30 fps,
cartoons animated "on twos". Upscaling those again is wasted GPU time, so the
video pipeline upscales only the unique frames and fills the rest with a link
(or copy) of the upscaled frame they duplicate. Frame count and timing stay
exactly the same.

Detection is cheap: each frame is reduced to a small grayscale thumbnail and
compared with the previous *kept* frame (not simply the previous frame, so a
slow fade or pan cannot creep past the threshold one tiny step at a time).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

# Thumbnail width used for comparison (height follows the aspect ratio).
THUMB_WIDTH = 96
# A frame is a duplicate when the mean absolute difference of the thumbnails
# (0..255 scale) is below MEAN_THRESHOLD and no thumbnail pixel moved more than
# MAX_THRESHOLD. The second test keeps small moving objects (a ball, a mouth,
# a subtitle change) from being swallowed by the average.
MEAN_THRESHOLD = 0.6
MAX_THRESHOLD = 10.0


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """cv2.imread that also works with non-ASCII Windows paths."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: Path, img: np.ndarray) -> bool:
    """cv2.imwrite that also works with non-ASCII Windows paths."""
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    tmp = Path(str(path) + ".tmp")
    buf.tofile(str(tmp))
    tmp.replace(path)
    return True


def thumbnail(img: np.ndarray, width: int = THUMB_WIDTH) -> np.ndarray:
    """Small float32 grayscale version of an image (BGR or gray input)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    tw = max(1, min(width, w))
    th = max(1, int(round(h * tw / w)))
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)


def frame_signature(path: Path, width: int = THUMB_WIDTH) -> Optional[np.ndarray]:
    img = read_image(path, cv2.IMREAD_GRAYSCALE)
    return None if img is None else thumbnail(img, width)


def is_near_duplicate(a: Optional[np.ndarray], b: Optional[np.ndarray],
                      mean_threshold: float = MEAN_THRESHOLD,
                      max_threshold: float = MAX_THRESHOLD) -> bool:
    """True when two thumbnails are close enough to reuse one upscaled frame."""
    if a is None or b is None or a.shape != b.shape:
        return False
    diff = np.abs(a - b)
    return float(diff.mean()) < mean_threshold and float(diff.max()) < max_threshold


def plan_unique(signatures: Sequence[Optional[np.ndarray]],
                mean_threshold: float = MEAN_THRESHOLD,
                max_threshold: float = MAX_THRESHOLD,
                max_run: int = 0) -> List[int]:
    """For every frame, the index of the frame whose upscaled image it should use.

    ref[i] == i means frame i is unique and must be upscaled. Frame 0 is always
    unique. A signature of None (unreadable frame) is always treated as unique.
    max_run > 0 limits how many frames in a row may reuse one kept frame.
    """
    ref: List[int] = []
    kept = -1
    run = 0
    for i, sig in enumerate(signatures):
        if (kept >= 0 and (max_run <= 0 or run < max_run)
                and is_near_duplicate(signatures[kept], sig, mean_threshold, max_threshold)):
            ref.append(kept)
            run += 1
        else:
            ref.append(i)
            kept, run = i, 0
    return ref


def find_duplicates(files: Sequence[Path], check: Optional[Callable[[], None]] = None,
                    **kw) -> List[int]:
    """plan_unique() for a list of frame files. check() is called now and then (cancel/pause)."""
    sigs = []
    for i, f in enumerate(files):
        if check and i % 16 == 0:
            check()
        sigs.append(frame_signature(f))
    return plan_unique(sigs, **kw)


def unique_indices(ref: Sequence[int]) -> List[int]:
    return [i for i, r in enumerate(ref) if r == i]


def output_positions(ref: Sequence[int]) -> List[int]:
    """For every frame, the 0-based position of its source among the unique frames.

    The AI engine only sees the unique frames, numbered 1..U, so frame i of the
    chunk is filled from upscaled file number output_positions(ref)[i] + 1.
    """
    pos_of = {}
    out = []
    for i, r in enumerate(ref):
        if r == i:
            pos_of[i] = len(pos_of)
        out.append(pos_of[r])
    return out
