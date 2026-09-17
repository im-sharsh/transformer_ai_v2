"""Ingestion metadata: file fingerprints, dataset IDs, and a registry of ingested files."""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file in chunks so large CSVs never need to fit in memory twice."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def make_dataset_id(path: str | Path, sha256: str) -> str:
    """Readable and content-based: same file content -> same ID."""
    return f"{Path(path).stem}_{sha256[:12]}"


@dataclass
class IngestionMetadata:
    dataset_id: str
    source: str
    file_name: str
    file_format: str
    file_size_bytes: int
    sha256: str
    ingested_at: str
    rows: int
    columns: int
    column_names: list
    dtypes: dict
    missing_cells: int
    load_seconds: float
    memory_mb: float
    pandas_version: str = pd.__version__
    python_version: str = platform.python_version()

    def to_dict(self) -> dict:
        return asdict(self)


class MetadataRegistry:
    """One JSON file per dataset plus an append-only registry.jsonl history."""

    def __init__(self, metadata_dir: str | Path):
        self.dir = Path(metadata_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.dir / "registry.jsonl"

    def save(self, meta: IngestionMetadata) -> Path:
        path = self.dir / f"{meta.dataset_id}.json"
        path.write_text(json.dumps(meta.to_dict(), indent=2))
        entry = {k: getattr(meta, k) for k in
                 ["dataset_id", "source", "sha256", "ingested_at", "rows", "columns"]}
        with open(self.registry_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return path

    def history(self) -> list[dict]:
        if not self.registry_file.exists():
            return []
        return [json.loads(line) for line in self.registry_file.read_text().splitlines() if line.strip()]

    def last_for_source(self, source: str | Path) -> dict | None:
        matches = [e for e in self.history() if e["source"] == str(Path(source).resolve())]
        return matches[-1] if matches else None
