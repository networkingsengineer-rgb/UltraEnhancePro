@echo off
REM ==================================================================
REM  UltraEnhance Pro - build script (run on the BUILD PC only)
REM  Produces:
REM    dist\UltraEnhancePro\UltraEnhancePro.exe   (app folder)
REM    dist\UltraEnhancePro_Portable.zip          (portable version)
REM    dist\UltraEnhancePro_Setup.exe             (installer, if Inno Setup 6 is installed)
REM  End users need none of this: Python, packages, FFmpeg and the AI
REM  models are all bundled inside the installer and the ZIP.
REM ==================================================================
setlocal enableextensions
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3.11"
) else (
    set "PY=python"
)

echo [1/6] Creating virtual environment...
if not exist .venv (
    %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail

echo [2/6] Installing Python packages...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements-dev.txt || goto :fail

echo [3/6] Downloading FFmpeg, Real-ESRGAN, RIFE and face models into vendor\ ...
python tools\fetch_deps.py || goto :fail

echo [4/6] Running unit tests...
python -m pytest -q tests || goto :fail

echo [5/6] Building the app with PyInstaller...
python tools\make_icon.py
if exist build rmdir /s /q build
if exist dist\UltraEnhancePro rmdir /s /q dist\UltraEnhancePro
pyinstaller --noconfirm --clean UltraEnhancePro.spec || goto :fail
copy /y README.md dist\UltraEnhancePro\README.txt >nul
copy /y licenses\THIRD_PARTY_NOTICES.txt dist\UltraEnhancePro\THIRD_PARTY_NOTICES.txt >nul

echo [6/6] Creating portable ZIP and installer...
if exist dist\UltraEnhancePro_Portable.zip del dist\UltraEnhancePro_Portable.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\UltraEnhancePro' -DestinationPath 'dist\UltraEnhancePro_Portable.zip' -CompressionLevel Optimal" || goto :fail

set "ISCC="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
if defined ISCC (
    "%ISCC%" installer\UltraEnhancePro.iss || goto :fail
) else (
    echo Inno Setup 6 not found - skipping installer. Get it free from https://jrsoftware.org/isdl.php
)

echo.
echo ===== BUILD COMPLETE =====
dir /b dist
exit /b 0

:fail
echo.
echo ***** BUILD FAILED - see the messages above *****
exit /b 1
