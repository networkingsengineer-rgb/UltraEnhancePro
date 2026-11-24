@echo off
REM Runs the unit tests (presets, scaling maths, tiling, FFmpeg commands, job queue).
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Creating test environment...
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    call .venv\Scripts\activate.bat
    python -m pip install -r requirements-dev.txt
)
python -m pytest -v tests
pause
