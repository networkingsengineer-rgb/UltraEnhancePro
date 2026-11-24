"""Paths, settings persistence and logging."""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_DIR = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "UltraEnhancePro"
LOG_FILE = APP_DIR / "log.txt"
SETTINGS_FILE = APP_DIR / "settings.json"
JOBS_DIR = APP_DIR / "jobs"


def resource_root() -> Path:
    """Folder that holds the bundled 'vendor' directory.

    * PyInstaller build: sys._MEIPASS (the _internal folder)
    * Running from source: the project root (parent of the package)
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def vendor_dir() -> Path:
    override = os.environ.get("ULTRAENHANCE_VENDOR")
    if override:
        return Path(override)
    return resource_root() / "vendor"


@dataclass
class Settings:
    gpu_id: int = -2              # -2 = auto, -1 = CPU, 0..n = GPU index
    tile_size: int = 0            # 0 = auto (from VRAM)
    threads: str = "1:2:2"        # Real-ESRGAN load:proc:save threads
    temp_dir: str = ""
    output_dir: str = ""
    default_preset: str = "1080p Full HD"
    language: str = "English"
    theme: str = "dark"
    jpg_quality: int = 95
    keep_exif: bool = True
    video_encoder: str = "auto"
    video_codec: str = "H.264"
    video_quality: str = "Balanced"
    video_container: str = "MP4"
    recent_files: list = field(default_factory=list)

    def resolved_output_dir(self) -> Path:
        p = Path(self.output_dir) if self.output_dir else Path.home() / "Pictures" / "UltraEnhancePro"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_temp_dir(self) -> Path:
        p = Path(self.temp_dir) if self.temp_dir else APP_DIR / "temp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except Exception:
            return cls()

    def save(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def setup_logging() -> logging.Logger:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ultraenhance")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    return logger


log = logging.getLogger("ultraenhance")
