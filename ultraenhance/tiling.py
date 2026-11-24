"""Tile-size selection for Real-ESRGAN so big outputs fit in limited VRAM.

Real-ESRGAN ncnn-vulkan processes the image in square tiles (-t). Memory
needed grows roughly with tile_size^2 * scale^2, so we pick a tile from the
available VRAM and the model's scale. The values below come from the
upstream recommendations (0 = auto is only safe on big GPUs).
"""
from __future__ import annotations

import math
from typing import List, Tuple

MIN_TILE = 32
MAX_TILE = 512

# (min VRAM in MB, tile size for a 4x model)
_VRAM_TABLE = (
    (12000, 512),
    (8000, 400),
    (6000, 320),
    (4000, 256),
    (3000, 192),
    (2000, 128),
    (1000, 64),
    (0, 32),
)


def safe_tile_size(vram_mb: int | None, scale: int = 4, cpu: bool = False) -> int:
    """Return a safe -t value. CPU mode uses a modest tile to bound RAM."""
    if cpu or not vram_mb:
        return 128
    tile = 32
    for min_mb, t in _VRAM_TABLE:
        if vram_mb >= min_mb:
            tile = t
            break
    # 2x/3x models use less memory per output pixel -> allow a bigger tile.
    if scale < 4:
        tile = int(tile * math.sqrt(4 / max(1, scale)))
    tile = max(MIN_TILE, min(MAX_TILE, tile))
    return tile - (tile % 16)  # multiples of 16 are friendliest to the shaders


def tile_grid(width: int, height: int, tile: int, overlap: int = 0) -> List[Tuple[int, int, int, int]]:
    """Split an image into tiles (x, y, w, h). Used for our own CPU-side
    processing (face restore / previews) and in tests to validate coverage."""
    if tile <= 0:
        return [(0, 0, width, height)]
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    step = tile - overlap
    tiles = []
    y = 0
    while True:
        h = min(tile, height - y)
        x = 0
        while True:
            w = min(tile, width - x)
            tiles.append((x, y, w, h))
            if x + w >= width:
                break
            x += step
        if y + h >= height:
            break
        y += step
    return tiles


def estimate_vram_mb(tile: int, scale: int = 4) -> int:
    """Rough VRAM estimate in MB for one tile (used for the UI hint)."""
    pixels_out = (tile * scale) ** 2
    return int(pixels_out * 4 * 3 * 12 / (1024 * 1024)) + 200
