from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trs.utils.io import read_yaml


@dataclass
class Settings:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        return cls(raw=read_yaml(path))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
