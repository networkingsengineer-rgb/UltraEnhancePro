"""Start the real main window off-screen and fail on any exception.

Run in CI on Windows before packaging so start-up crashes never ship.
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
home = Path(tempfile.mkdtemp())
os.environ["USERPROFILE"] = str(home)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

errors = []
sys.excepthook = lambda t, e, tb: errors.append("".join(traceback.format_exception(t, e, tb)))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication(sys.argv)
from ultraenhance.ui.main_window import MainWindow  # noqa: E402


def build(label):
    try:
        w = MainWindow()
        w.show()
        QTimer.singleShot(3000, app.quit)
        app.exec()
        w.settings.save()
        w.close()
    except Exception:
        errors.append(f"{label}:\n" + traceback.format_exc())


build("fresh start")
# Second start with a settings file holding wrong types (old versions / hand edits)
(home / "UltraEnhancePro").mkdir(parents=True, exist_ok=True)
(home / "UltraEnhancePro" / "settings.json").write_text(
    '{"threads": 4, "tile_size": "256", "jpg_quality": "95", "keep_exif": 1}', encoding="utf-8")
build("start with old settings")
build("restart after save")

if errors:
    print("SMOKE TEST FAILED")
    print("\n".join(errors))
    sys.exit(1)
print("SMOKE TEST PASSED")
