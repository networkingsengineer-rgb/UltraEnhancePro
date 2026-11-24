# PyInstaller spec - one-folder build (fast start, easy to zip as portable).
# Build with:  pyinstaller --noconfirm UltraEnhancePro.spec
from pathlib import Path

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "vendor"), "vendor"),
    (str(ROOT / "licenses"), "licenses"),
    (str(ROOT / "assets"), "assets"),
]

a = Analysis(
    [str(ROOT / "run_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["pillow_heif", "onnxruntime", "psutil", "cv2"],
    excludes=["tkinter", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
              "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UltraEnhancePro",
    console=False,
    icon=str(ROOT / "assets" / "icon.ico"),
    version=str(ROOT / "tools" / "version_info.txt"),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="UltraEnhancePro")
