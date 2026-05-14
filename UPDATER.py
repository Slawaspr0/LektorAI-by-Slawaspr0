from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APP_PACKAGES_DIR = APP_DIR / "packages"
if APP_PACKAGES_DIR.is_dir():
    sys.path.insert(0, str(APP_PACKAGES_DIR))

from app.updater.core import run_updater_cli


if __name__ == "__main__":
    raise SystemExit(run_updater_cli())

