"""Format-aware dataset loader. Reads raw files; never writes to them."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .metadata import IngestionMetadata, MetadataRegistry, file_sha256, make_dataset_id

logger = logging.getLogger("INGESTION")

SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "jsonl",
}
# Future connectors (Excel, SQLite, PostgreSQL, ...) register a reader here.
READERS = {
    "csv": lambda p, **kw: pd.read_csv(p, low_memory=False, **kw),
    "parquet": lambda p, **kw: pd.read_parquet(p, **kw),
    "json": lambda p, **kw: pd.read_json(p, **kw),
    "jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
}


class UnsupportedFormatError(ValueError):
    pass


class DatasetIntegrityError(RuntimeError):
    pass


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    metadata: IngestionMetadata


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}")
    return SUPPORTED_FORMATS[suffix]


def load_dataset(path: str | Path,
                 metadata_dir: str | Path | None = None,
                 expected_sha256: str | None = None,
                 save_metadata: bool = True,
                 **read_kwargs) -> LoadedDataset:
    """Load a raw dataset and record ingestion metadata.

    Values are loaded exactly as pandas reads them; no cleaning or type coercion
    happens here. Pass read_kwargs (e.g. dtype=str) only for deliberate experiments.
    """
    path = Path(path).resolve()
    t_start = time.time()
    logger.info("START load %s", path.name)

    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size == 0:
        raise DatasetIntegrityError(f"{path.name} is empty (0 bytes)")

    fmt = detect_format(path)
    sha = file_sha256(path)

    if expected_sha256 and sha != expected_sha256:
        raise DatasetIntegrityError(
            f"{path.name}: hash {sha[:12]} does not match expected {expected_sha256[:12]}")

    registry = MetadataRegistry(metadata_dir or path.parent / "_metadata")
    previous = registry.last_for_source(path)
    if previous and previous["sha256"] != sha:
        logger.warning("Raw file %s changed since last ingestion (%s -> %s). "
                       "Treat this as a new dataset version.",
                       path.name, previous["sha256"][:12], sha[:12])

    try:
        t_read = time.time()
        df = READERS[fmt](path, **read_kwargs)
        read_seconds = time.time() - t_read
    except Exception as exc:  # unreadable or malformed file
        raise DatasetIntegrityError(f"Could not read {path.name} as {fmt}: {exc}") from exc

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DatasetIntegrityError(f"{path.name} loaded with shape {df.shape}")

    meta = IngestionMetadata(
        dataset_id=make_dataset_id(path, sha),
        source=str(path),
        file_name=path.name,
        file_format=fmt,
        file_size_bytes=size,
        sha256=sha,
        ingested_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        missing_cells=int(df.isna().sum().sum()),
        load_seconds=round(read_seconds, 3),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    )
    if save_metadata:
        saved = registry.save(meta)
        logger.info("OUTPUT metadata saved to %s", saved)

    logger.info("Loaded %d rows x %d columns (%s, %.1f MB) in %.1fs",
                meta.rows, meta.columns, fmt, meta.memory_mb, time.time() - t_start)
    if meta.missing_cells:
        logger.warning("%d missing cells detected (not modified)", meta.missing_cells)
    return LoadedDataset(df=df, metadata=meta)
