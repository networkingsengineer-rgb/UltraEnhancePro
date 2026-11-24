"""Launcher used by PyInstaller and for running from source: python run_app.py"""
import sys

from ultraenhance.main import main

if __name__ == "__main__":
    sys.exit(main())
