"""Role-aware data profiler. Measures the data; never modifies it."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.schema_detector import SchemaReport

logger = logging.getLogger("PROFILING")

NUMERIC_ROLES = {"numeric"}
CATEGORICAL_ROLES = {"categorical", "categorical_code", "boolean", "target"}
DATETIME_ROLES = {"datetime", "datetime_epoch"}
ID_ROLES = {"record_id", "entity_id", "row_index"}


def _py(value):
    """Convert numpy/pandas scalars to JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return None if pd.isna(value) else str(value)
    if isinstance(value, float):
        return None if np.isnan(value) else round(value, 6)
    return value


def parse_datetime_column(s: pd.Series, role: str) -> pd.Series:
    """Parse text or Unix-seconds columns into datetimes; unparseable values become NaT."""
    if role == "datetime_epoch":
        return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="s", errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    parsed = pd.to_datetime(s, errors="coerce", format="ISO8601")
    retry = parsed.isna() & s.notna()                 # re-parse only the non-ISO values (keeps it fast)
    if retry.any():
        parsed.loc[retry] = pd.to_datetime(s[retry].astype(str), errors="coerce", format="mixed")
    return parsed


@dataclass
class ColumnProfile:
    name: str
    role: str
    dtype: str
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class DatasetProfile:
    dataset_id: str
    profiled_at: str
    seconds: float
    overview: dict
    target: dict
    entity: dict
    columns: dict
    warnings: list

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        j = out_dir / f"profile_{self.dataset_id}.json"
        j.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        m = out_dir / f"profile_{self.dataset_id}.md"
        m.write_text(self.to_markdown())
        return j, m

    def to_markdown(self) -> str:
        o = self.overview
        out = [f"# Data profile: {self.dataset_id}", "",
               f"Profiled {self.profiled_at} in {self.seconds}s", "",
               "## Overview", "",
               f"- Rows: {o['rows']:,}; columns: {o['columns']}",
               f"- Memory: {o['memory_mb']} MB; missing cells: {o['missing_cells']:,} ({o['missing_cells_pct']}%)",
               f"- Exact duplicate rows: {o['exact_duplicate_rows']:,}; duplicates ignoring ID columns: "
               f"{o['duplicate_rows_ignoring_ids']:,}", ""]
        out += ["## Warnings", ""] + ([f"- {w}" for w in self.warnings] or ["- None"]) + [""]
        if self.target:
            t = self.target
            out += ["## Target", "", f"- Column `{t['column']}`; positive rate {t['positive_rate']:.4%} "
                    f"({t['positives']:,} of {t['count']:,})", ""]
        if self.entity:
            e = self.entity
            out += ["## Entity", "", f"- `{e['column']}`: {e['entities']:,} entities; transactions per entity "
                    f"min {e['per_entity']['min']}, median {e['per_entity']['median']}, max {e['per_entity']['max']}"]
            if "median_gap_seconds" in e:
                out.append(f"- Median time between an entity's consecutive transactions: "
                           f"{e['median_gap_seconds'] / 3600:.1f} h")
            out.append("")
        num = [c for c in self.columns.values() if c.role in NUMERIC_ROLES]
        if num:
            out += ["## Numeric columns", "",
                    "| Column | Missing % | Mean | Median | Std | Min | Max | Skew | Kurtosis | Zeros | Negatives | IQR outliers | Extreme (3×IQR) |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in num:
                s = c.stats
                out.append(f"| {c.name} | {c.missing_pct} | {s['mean']} | {s['median']} | {s['std']} | {s['min']} | "
                           f"{s['max']} | {s['skewness']} | {s['kurtosis']} | {s['zero_count']:,} | "
                           f"{s['negative_count']:,} | {s['outliers_iqr_1_5']:,} | {s['outliers_iqr_3_0']:,} |")
            out.append("")
        cat = [c for c in self.columns.values() if c.role in CATEGORICAL_ROLES]
        if cat:
            out += ["## Categorical columns", "",
                    "| Column | Role | Missing % | Unique | Top values | Rare categories | Rows in rare |",
                    "|---|---|---|---|---|---|---|"]
            for c in cat:
                s = c.stats
                top = ", ".join(f"{k} ({v:.1%})" for k, v in s["top_values"].items())
                out.append(f"| {c.name} | {c.role} | {c.missing_pct} | {c.unique_count:,} | {top} | "
                           f"{s['rare_category_count']:,} | {s['rare_rows_pct']}% |")
            out.append("")
        dts = [c for c in self.columns.values() if c.role in DATETIME_ROLES]
        if dts:
            out += ["## Datetime columns", "",
                    "| Column | Min | Max | Invalid | Span (days) | Largest gap | Median rows/day |",
                    "|---|---|---|---|---|---|---|"]
            for c in dts:
                s = c.stats
                out.append(f"| {c.name} | {s['min']} | {s['max']} | {s['invalid_count']:,} | {s['span_days']} | "
                           f"{s['largest_gap']} | {s['median_rows_per_day']} |")
            out.append("")
            for c in dts:
                if c.stats.get("monthly"):
                    out += [f"### Monthly volume: `{c.name}`", "", "| Month | Rows | Target rate |", "|---|---|---|"]
                    for month, v in c.stats["monthly"].items():
                        rate = "" if v.get("target_rate") is None else f"{v['target_rate']:.3%}"
                        out.append(f"| {month} | {v['rows']:,} | {rate} |")
                    out.append("")
        ids = [c for c in self.columns.values() if c.role in ID_ROLES]
        if ids:
            out += ["## Identifier columns", "", "| Column | Role | Unique | Duplicated values |", "|---|---|---|---|"]
            for c in ids:
                out.append(f"| {c.name} | {c.role} | {c.unique_count:,} | {c.stats.get('duplicated_values', 0):,} |")
            out.append("")
        return "\n".join(out) + "\n"


class DataProfiler:
    def __init__(self, top_k: int = 5, rare_threshold: float = 0.001, mask_pii: bool = True,
                 skew_warning: float = 2.0, missing_warning_pct: float = 5.0):
        self.top_k = top_k
        self.rare_threshold = rare_threshold
        self.mask_pii = mask_pii
        self.skew_warning = skew_warning
        self.missing_warning_pct = missing_warning_pct

    def profile(self, df: pd.DataFrame, schema: SchemaReport) -> DatasetProfile:
        t0 = time.time()
        logger.info("START profiling %s (%d rows x %d columns)", schema.dataset_id, *df.shape)
        warnings: list[str] = []
        target_col = schema.target
        target = pd.to_numeric(df[target_col], errors="coerce") if target_col else None

        columns = {}
        for name, cs in schema.columns.items():
            columns[name] = self._profile_column(df[name], cs, target)
            warnings += [f"`{name}`: {w}" for w in columns[name].warnings]

        id_cols = [c for c, cs in schema.columns.items() if cs.role in ("record_id", "row_index")]
        exact_dupes = int(df.duplicated().sum())
        dupes_no_ids = int(df.drop(columns=id_cols).duplicated().sum()) if id_cols else exact_dupes
        if dupes_no_ids:
            warnings.append(f"{dupes_no_ids:,} rows are identical apart from ID columns (possible repeated records)")
        missing = int(df.isna().sum().sum())
        overview = {
            "rows": len(df), "columns": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 1),
            "missing_cells": missing, "missing_cells_pct": round(100 * missing / max(df.size, 1), 4),
            "exact_duplicate_rows": exact_dupes, "duplicate_rows_ignoring_ids": dupes_no_ids,
            "columns_by_role": schema.by_role(),
        }

        target_info = {}
        if target_col:
            positives = int((target == 1).sum())
            target_info = {"column": target_col, "count": int(target.notna().sum()), "positives": positives,
                           "positive_rate": positives / max(int(target.notna().sum()), 1)}
            if target_info["positive_rate"] < 0.05:
                warnings.append(f"Target is highly imbalanced ({target_info['positive_rate']:.3%} positive): "
                                "use precision/recall/PR-AUC, not accuracy")

        entity_info = self._profile_entity(df, schema)
        prof = DatasetProfile(dataset_id=schema.dataset_id, profiled_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              seconds=round(time.time() - t0, 1), overview=overview, target=target_info,
                              entity=entity_info, columns=columns, warnings=warnings)
        logger.info("Profiled %d columns, %d warnings in %.1fs", len(columns), len(warnings), prof.seconds)
        return prof

    # ---------- column profiles ----------
    def _profile_column(self, s: pd.Series, cs, target) -> ColumnProfile:
        missing = int(s.isna().sum())
        cp = ColumnProfile(name=cs.name, role=cs.role, dtype=str(s.dtype), count=int(s.notna().sum()),
                           missing_count=missing, missing_pct=round(100 * missing / max(len(s), 1), 4),
                           unique_count=int(s.nunique(dropna=True)))
        if cp.missing_pct > self.missing_warning_pct:
            cp.warnings.append(f"{cp.missing_pct}% missing")
        if cs.role in NUMERIC_ROLES:
            self._numeric(s, cp)
        elif cs.role in CATEGORICAL_ROLES:
            self._categorical(s, cp, masked=self.mask_pii and cs.pii_type is not None)
        elif cs.role in DATETIME_ROLES:
            self._datetime(s, cs, cp, target)
        elif cs.role in ID_ROLES:
            cp.stats["duplicated_values"] = int(s.dropna().duplicated().sum())
            if cs.role == "record_id" and cp.stats["duplicated_values"]:
                cp.warnings.append(f"{cp.stats['duplicated_values']:,} repeated values in a record ID column")
        elif cs.role == "constant":
            cp.warnings.append("constant column carries no information")
        return cp

    def _numeric(self, s: pd.Series, cp: ColumnProfile):
        x = pd.to_numeric(s, errors="coerce")
        invalid = int(x.isna().sum() - s.isna().sum())
        v = x.dropna()
        if v.empty:
            cp.warnings.append("no parseable numeric values")
            return
        q = v.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        iqr = q[0.75] - q[0.25]
        std = v.std()
        cp.stats = {
            "mean": _py(v.mean()), "median": _py(q[0.5]), "std": _py(std),
            "min": _py(v.min()), "max": _py(v.max()),
            "quantiles": {f"p{int(k * 100)}": _py(val) for k, val in q.items()},
            "skewness": _py(v.skew()), "kurtosis": _py(v.kurt()),
            "zero_count": int((v == 0).sum()), "negative_count": int((v < 0).sum()),
            "non_numeric_values": invalid,
            "outliers_iqr_1_5": int(((v < q[0.25] - 1.5 * iqr) | (v > q[0.75] + 1.5 * iqr)).sum()),
            "outliers_iqr_3_0": int(((v < q[0.25] - 3 * iqr) | (v > q[0.75] + 3 * iqr)).sum()),
            "outliers_zscore_3": int((((v - v.mean()) / std).abs() > 3).sum()) if std else 0,
            "duplicate_value_count": int(v.duplicated().sum()),
        }
        if invalid:
            cp.warnings.append(f"{invalid:,} values could not be parsed as numbers")
        if cp.stats["skewness"] is not None and abs(cp.stats["skewness"]) > self.skew_warning:
            hint = "; log transform is an option" if cp.stats["min"] >= 0 else ""
            cp.warnings.append(f"highly skewed (skewness {cp.stats['skewness']:.2f}){hint}")

    def _categorical(self, s: pd.Series, cp: ColumnProfile, masked: bool):
        counts = s.value_counts(dropna=True)
        total = max(int(counts.sum()), 1)
        freq = counts / total
        rare = freq[freq < self.rare_threshold]
        top = freq.head(self.top_k)
        if masked:
            top_values = {f"<masked #{i + 1}>": round(float(p), 6) for i, p in enumerate(top)}
        else:
            top_values = {str(k)[:40]: round(float(p), 6) for k, p in top.items()}
        text = s.dropna().astype(str)
        stripped_lower = text.str.strip().str.lower()
        variants = int(text.nunique() - stripped_lower.nunique())
        cp.stats = {
            "top_values": top_values, "values_masked": masked,
            "rare_threshold": self.rare_threshold, "rare_category_count": int(len(rare)),
            "rare_rows_pct": round(100 * float(rare.sum()), 4),
            "case_or_whitespace_variants": variants,
        }
        if variants:
            cp.warnings.append(f"{variants} categories differ only by case/whitespace (e.g. 'UPI' vs 'upi')")
        if len(rare) > 0 and cp.role != "target":
            cp.warnings.append(f"{len(rare):,} rare categories (<{self.rare_threshold:.1%} each) "
                               f"covering {cp.stats['rare_rows_pct']}% of rows")

    def _datetime(self, s: pd.Series, cs, cp: ColumnProfile, target):
        dt = parse_datetime_column(s, cs.role)
        invalid = int(dt.isna().sum() - s.isna().sum())
        v = dt.dropna()
        if v.empty:
            cp.warnings.append("no parseable datetimes")
            return
        uniq = np.sort(v.unique())
        gaps = pd.Series(np.diff(uniq)) if len(uniq) > 1 else pd.Series(dtype="timedelta64[ns]")
        per_day = v.dt.floor("D").value_counts()
        cp.stats = {
            "min": _py(v.min()), "max": _py(v.max()), "invalid_count": invalid,
            "span_days": round((v.max() - v.min()).total_seconds() / 86400, 1),
            "largest_gap": _py(gaps.max()) if len(gaps) else None,
            "median_rows_per_day": _py(float(per_day.median())),
            "days_with_no_rows": int(((v.max().normalize() - v.min().normalize()).days + 1) - len(per_day)),
            "future_values": int((v > pd.Timestamp.now()).sum()),
        }
        if invalid:
            cp.warnings.append(f"{invalid:,} unparseable timestamps")
        if cp.stats["future_values"]:
            cp.warnings.append(f"{cp.stats['future_values']:,} timestamps are in the future")
        if cs.name and cs.pii_type != "birth_date":
            month = dt.dt.to_period("M").astype(str)
            frame = pd.DataFrame({"month": month, "t": target if target is not None else np.nan})[dt.notna()]
            grouped = frame.groupby("month")["t"].agg(["size", "mean"])
            cp.stats["monthly"] = {m: {"rows": int(r["size"]),
                                       "target_rate": None if target is None else _py(r["mean"])}
                                   for m, r in grouped.iterrows()}

    def _profile_entity(self, df: pd.DataFrame, schema: SchemaReport) -> dict:
        entities = schema.by_role().get("entity_id", [])
        if not entities:
            return {}
        col = entities[0]
        per = df[col].value_counts()
        info = {"column": col, "entities": int(per.size),
                "per_entity": {"min": int(per.min()), "median": _py(float(per.median())), "max": int(per.max())}}
        times = [c for c, cs in schema.columns.items() if cs.role == "datetime" and cs.pii_type != "birth_date"]
        if times:
            dt = parse_datetime_column(df[times[0]], "datetime")
            frame = pd.DataFrame({"e": df[col], "t": dt}).dropna().sort_values(["e", "t"])
            gaps = frame.groupby("e")["t"].diff().dropna().dt.total_seconds()
            if len(gaps):
                info.update({"time_column": times[0], "median_gap_seconds": _py(float(gaps.median())),
                             "p10_gap_seconds": _py(float(gaps.quantile(0.1))),
                             "zero_second_gaps": int((gaps == 0).sum())})
        return info


def profile_dataset(df: pd.DataFrame, schema: SchemaReport, **kwargs) -> DatasetProfile:
    return DataProfiler(**kwargs).profile(df, schema)
