"""Supported input and output file formats."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

PHOTO_EXTS = (
    ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".bmp", ".dib",
    ".tif", ".tiff", ".heic", ".heif", ".gif", ".tga", ".ppm", ".pgm",
)
VIDEO_EXTS = (
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".3gp", ".ogv", ".vob",
)

PHOTO_OUTPUT_FORMATS = {
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tiff",
}
VIDEO_CONTAINERS = {
    "MP4": ".mp4",
    "MKV": ".mkv",
    "MOV": ".mov",
    "WEBM": ".webm",
}


def is_photo(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in PHOTO_EXTS


def is_video(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def is_supported(path: str | os.PathLike) -> bool:
    return is_photo(path) or is_video(path)


def _pattern(exts: Iterable[str]) -> str:
    return " ".join("*" + e for e in exts)


def dialog_filter(kind: str = "all") -> str:
    """Qt file-dialog filter string. kind = 'all' | 'photo' | 'video'."""
    photos = f"Photos ({_pattern(PHOTO_EXTS)})"
    videos = f"Videos ({_pattern(VIDEO_EXTS)})"
    allm = f"All supported media ({_pattern(PHOTO_EXTS + VIDEO_EXTS)})"
    if kind == "photo":
        return f"{photos};;All files (*.*)"
    if kind == "video":
        return f"{videos};;All files (*.*)"
    return f"{allm};;{photos};;{videos};;All files (*.*)"


def expand_paths(paths: Iterable[str | os.PathLike], kind: str = "all") -> List[str]:
    """Expand dropped files/folders into a sorted, de-duplicated list of media files."""
    check = {"photo": is_photo, "video": is_video}.get(kind, is_supported)
    found: list[str] = []
    seen: set[str] = set()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            items = sorted(x for x in p.rglob("*") if x.is_file())
        elif p.is_file():
            items = [p]
        else:
            continue
        for item in items:
            if check(item):
                key = str(item.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    found.append(str(item))
    return found
