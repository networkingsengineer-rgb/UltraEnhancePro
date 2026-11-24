"""Application entry point."""
from __future__ import annotations

import multiprocessing
import os
import sys
import traceback


def _excepthook(exc_type, exc, tb):
    from .config import log
    log.critical("Unhandled error:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from .job_queue import friendly_error
        if QApplication.instance():
            QMessageBox.critical(None, "UltraEnhance Pro",
                                 "Something went wrong:\n\n" + friendly_error(exc) +
                                 "\n\nDetails were written to the log file (Tools > Open log file).")
    except Exception:
        pass


def main() -> int:
    multiprocessing.freeze_support()
    from .config import setup_logging
    log = setup_logging()
    from . import __version__
    log.info("UltraEnhance Pro %s starting (Python %s)", __version__, sys.version.split()[0])
    sys.excepthook = _excepthook

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("UltraEnhance Pro")
    app.setOrganizationName("UltraEnhance")

    from .ui.main_window import MainWindow
    win = MainWindow()
    win.show()
    # Files passed on the command line / "Open with" from Explorer
    paths = [a for a in sys.argv[1:] if os.path.exists(a)]
    if paths:
        win.add_paths(paths)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
