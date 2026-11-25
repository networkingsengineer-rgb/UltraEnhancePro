"""Video form smoke test: modes, new widgets, old saved options (skipped where Qt cannot load)."""
import pytest

try:
    from PySide6.QtWidgets import QApplication
    from ultraenhance.ui.main_window import MainWindow
except Exception as e:  # pragma: no cover - e.g. Linux without libEGL
    pytest.skip(f"Qt widgets unavailable: {e}", allow_module_level=True)

from ultraenhance.presets import VIDEO_RESOLUTION_PRESETS
from ultraenhance.video import VideoOptions


@pytest.fixture(scope="module")
def win():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    yield w
    w.close()


def test_mode_sets_widgets_and_edit_switches_to_custom(win):
    win.v_mode.setCurrentIndex(win.v_mode.findData("Restore old video"))
    o = win.video_form()
    assert o.cleanup == "strong" and o.face_enhance and o.auto_color and o.sharpen and o.denoise
    win.v_sharpen.setChecked(False)
    assert win.v_mode.currentData() == "Custom"
    win.v_mode.setCurrentIndex(win.v_mode.findData("Best quality (slow)"))
    o = win.video_form()
    assert o.model == "Best quality (slow)" and o.quality == "Best" and o.cleanup == "light" and o.face_enhance


def test_resolution_list_and_old_presets(win):
    assert [win.v_preset.itemData(i) for i in range(win.v_preset.count())] == list(VIDEO_RESOLUTION_PRESETS)
    win.load_video_form(VideoOptions(preset="4K (2160p)"))
    assert win.v_preset.currentData() == "4K (2160p UHD)"
    assert "H.265/HEVC" in win._video_summary(VideoOptions(preset="8K (4320p)", codec="H.264"))
