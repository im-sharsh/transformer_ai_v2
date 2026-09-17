"""Temporal train/validation/test assignment, defined on row IDs so every condition uses the same rows."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class SplitConfig:
    time_column: str
    train_end: str                               # exclusive: rows before this date are training rows
    test_sources: tuple = ()                     # rows from these files are test rows ...
    test_start: str | None = None                # ... or rows at/after this date (if no test_sources)
    source_column: str = "_source_file"
    strategy: str = "temporal"                   # temporal | random
    random_fractions: tuple = (0.7, 0.15, 0.15)
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def assign_splits(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    if cfg.strategy == "random":
        rng = pd.Series(pd.util.hash_pandas_object(df["_row_id"], index=False).values % 10_000 / 10_000,
                        index=df.index)
        a, b, _ = cfg.random_fractions
        return pd.Series(pd.cut(rng, [-0.1, a, a + b, 1.1], labels=["train", "validation", "test"]).astype(str),
                         index=df.index)
    t = pd.to_datetime(df[cfg.time_column], errors="coerce")
    if cfg.test_sources:
        is_test = df[cfg.source_column].isin(cfg.test_sources)
    elif cfg.test_start:
        is_test = t >= pd.Timestamp(cfg.test_start)
    else:
        raise ValueError("Give test_sources or test_start")
    split = pd.Series("validation", index=df.index, dtype="object")
    split[t < pd.Timestamp(cfg.train_end)] = "train"
    split[is_test] = "test"
    split[t.isna() & ~is_test] = "unassigned"
    return split


def split_report(df: pd.DataFrame, split: pd.Series, target: str, time_column: str) -> dict:
    t = pd.to_datetime(df[time_column], errors="coerce")
    out = {}
    for name in ["train", "validation", "test", "unassigned"]:
        m = split == name
        if not m.any():
            continue
        y = pd.to_numeric(df.loc[m, target], errors="coerce")
        out[name] = {"rows": int(m.sum()), "positives": int((y == 1).sum()),
                     "positive_rate": round(float((y == 1).mean()), 6),
                     "time_min": str(t[m].min()), "time_max": str(t[m].max())}
    if {"train", "validation"} <= out.keys():
        assert pd.Timestamp(out["train"]["time_max"]) < pd.Timestamp(out["validation"]["time_min"]), \
            "training rows overlap the validation period"
    return out
