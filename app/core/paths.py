from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    app_dir: Path

    @property
    def config_path(self) -> Path:
        return self.app_dir / "config.json"

    @property
    def logs_dir(self) -> Path:
        return self.app_dir / "logs"

    @property
    def runtime_engines_dir(self) -> Path:
        return self.app_dir / "engines"

    @property
    def temp_dir(self) -> Path:
        return self.app_dir / "temp"

    @property
    def app_packages_dir(self) -> Path:
        return self.app_dir / "packages"

    def engine_dir(self, engine_id: str) -> Path:
        return self.runtime_engines_dir / engine_id


def build_paths(app_dir: Path) -> AppPaths:
    return AppPaths(app_dir=app_dir.resolve())
