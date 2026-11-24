import pytest

from ultraenhance.presets import (RESOLUTION_PRESETS, compute_geometry, even, fill_size, fit_size,
                                  plan_passes, required_factor, target_box)


def test_all_presets_present():
    assert list(RESOLUTION_PRESETS) == ["720p HD", "1080p Full HD", "2K (1440p)", "4K (2160p)", "8K (4320p)"]
    assert RESOLUTION_PRESETS["4K (2160p)"] == (3840, 2160)
    assert RESOLUTION_PRESETS["8K (4320p)"] == (7680, 4320)


def test_target_box_portrait_swaps():
    assert target_box("1080p Full HD", 1920, 1080) == (1920, 1080)
    assert target_box("1080p Full HD", 1080, 1920) == (1080, 1920)


def test_fit_keeps_aspect():
    assert fit_size(640, 480, 1920, 1080) == (1440, 1080)
    assert fit_size(1280, 720, 3840, 2160) == (3840, 2160)


def test_fill_covers_box():
    w, h = fill_size(640, 480, 1920, 1080)
    assert w >= 1920 and h >= 1080 and (w, h) == (1920, 1440)


def test_geometry_fit():
    g = compute_geometry(640, 480, "1080p Full HD", mode="fit")
    assert (g.canvas_w, g.canvas_h) == (1440, 1080)
    assert (g.offset_x, g.offset_y) == (0, 0)


def test_geometry_pad_centres():
    g = compute_geometry(640, 480, "1080p Full HD", mode="pad")
    assert (g.canvas_w, g.canvas_h) == (1920, 1080)
    assert (g.scaled_w, g.scaled_h) == (1440, 1080)
    assert g.offset_x == 240 and g.offset_y == 0


def test_geometry_fill_crops():
    g = compute_geometry(640, 480, "1080p Full HD", mode="fill")
    assert (g.canvas_w, g.canvas_h) == (1920, 1080)
    assert (g.scaled_w, g.scaled_h) == (1920, 1440)
    assert g.offset_y == -180


def test_geometry_scale():
    g = compute_geometry(500, 333, scale=3)
    assert (g.canvas_w, g.canvas_h) == (1500, 999)
    g = compute_geometry(500, 333, scale=3, even_dims=True)
    assert g.canvas_w % 2 == 0 and g.canvas_h % 2 == 0


def test_even():
    assert even(1081) in (1080, 1082)
    assert even(1) == 2
    assert even(999) % 2 == 0


def test_bad_mode():
    with pytest.raises(ValueError):
        compute_geometry(100, 100, "720p HD", mode="stretch")
    with pytest.raises(ValueError):
        compute_geometry(100, 100)


def test_required_factor():
    g = compute_geometry(960, 540, "4K (2160p)")
    assert required_factor(960, 540, g) == pytest.approx(4.0)


@pytest.mark.parametrize("factor,native,expected", [
    (1.0, (4,), []),
    (0.5, (4,), []),
    (1.5, (4,), [4]),
    (4.0, (4,), [4]),
    (8.0, (4,), [4, 4]),
    (6.0, (4,), [4, 4]),
    (2.0, (2, 3, 4), [2]),
    (3.0, (2, 3, 4), [3]),
    (5.0, (2, 3, 4), [2, 3]),
    (8.0, (2, 3, 4), [2, 4]),
])
def test_plan_passes(factor, native, expected):
    res = plan_passes(factor, native)
    assert sorted(res) == sorted(expected)


def test_plan_passes_cap():
    assert plan_passes(1000, (4,), max_passes=3) == [4, 4, 4]
