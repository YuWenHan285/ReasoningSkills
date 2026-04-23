from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import orjson
import yaml


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_json(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = ensure_parent(path)
    with open(p, "wb") as f:
        for record in records:
            f.write(orjson.dumps(record))
            f.write(b"\n")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    p = ensure_parent(path)
    with open(p, "ab") as f:
        f.write(orjson.dumps(record))
        f.write(b"\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(orjson.loads(line))
    return records
