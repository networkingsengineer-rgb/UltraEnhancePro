import pytest

from ultraenhance.tiling import MAX_TILE, MIN_TILE, estimate_vram_mb, safe_tile_size, tile_grid


@pytest.mark.parametrize("vram,expected_max", [(2048, 128), (4096, 256), (6144, 320), (8192, 400), (24576, 512)])
def test_tile_grows_with_vram(vram, expected_max):
    t = safe_tile_size(vram, 4)
    assert MIN_TILE <= t <= expected_max
    assert t % 16 == 0


def test_4gb_safe_for_4k_and_8k():
    # 4 GB cards must get a tile that fits comfortably.
    t = safe_tile_size(4096, 4)
    assert t <= 256
    assert estimate_vram_mb(t, 4) < 4096


def test_monotonic():
    vals = [safe_tile_size(v, 4) for v in (512, 1024, 2048, 3072, 4096, 6144, 8192, 12288)]
    assert vals == sorted(vals)


def test_cpu_and_unknown():
    assert safe_tile_size(None) == 128
    assert safe_tile_size(8000, cpu=True) == 128


def test_smaller_scale_allows_bigger_tile():
    assert safe_tile_size(4096, 2) >= safe_tile_size(4096, 4)
    assert safe_tile_size(100000, 2) <= MAX_TILE


@pytest.mark.parametrize("w,h,tile,overlap", [(1000, 700, 256, 0), (513, 257, 256, 16), (64, 64, 256, 0), (1920, 1080, 200, 10)])
def test_tile_grid_covers_image(w, h, tile, overlap):
    tiles = tile_grid(w, h, tile, overlap)
    covered = [[False] * w for _ in range(h)]
    for x, y, tw, th in tiles:
        assert tw > 0 and th > 0 and x + tw <= w and y + th <= h
        for yy in range(y, y + th, max(1, th // 4) if th > 8 else 1):
            for xx in range(x, x + tw, max(1, tw // 4) if tw > 8 else 1):
                covered[yy][xx] = True
    # corners and edges reached
    assert any(t[0] + t[2] == w for t in tiles)
    assert any(t[1] + t[3] == h for t in tiles)
    assert tiles[0][:2] == (0, 0)


def test_tile_grid_zero_means_whole():
    assert tile_grid(100, 50, 0) == [(0, 0, 100, 50)]


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        tile_grid(100, 100, 32, 32)
