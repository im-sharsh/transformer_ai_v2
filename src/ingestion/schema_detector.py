"""Automatic schema detection.

Infers each column's role from its values and (as supporting evidence) its name.
No dataset-specific column names are hard-coded; manual overrides are supported.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("SCHEMA")

# Name patterns are *supporting evidence* only; value checks decide where possible.
NAME_HINTS = {
    "target": r"(^|_)(is_fraud|fraud_label|fraud|label|target|y)$",
    "id": r"(^|_)(id|uuid|guid|key|num|number|no)$|^id_",
    "entity": r"(customer|client|user|account|acct|card|cc|device|member|merchant_id)",
    "code": r"(^|_)(zip|zipcode|postal|postcode|pincode|pin|mcc|code)$",
    "epoch": r"(unix|epoch|timestamp|time|ts)",
    "geo": r"(^|_)(lat|long|lon|lng|latitude|longitude)$",
    "leak": r"(fraud|chargeback|dispute|label|outcome|confirmed)",
}
PII_NAME_HINTS = {
    "person_name": r"^(first|last|full|given|sur)_?name$|^(first|last|surname)$|^name$|(customer|cardholder|holder|user)_?name",
    "email": r"e_?mail",
    "phone": r"(phone|mobile|telephone)",
    "address": r"(street|address|addr)",
    "birth_date": r"(^dob$|birth)",
    "card_number": r"(cc_?num|card_?(num|no|number)|(^|_)pan$)",
    "account_number": r"(account|acct)_?(id|num|number|no)$|iban",
    "government_id": r"(ssn|aadhaar|passport|national_id|tax_id)",
    "location": r"(^|_)(city|zip|zipcode|postal|lat|long|lon|lng|latitude|longitude)$",
}
EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$")
PHONE_RE = re.compile(r"^\+?[\d\s()-]{8,16}$")
HEX_ID_RE = re.compile(r"^[0-9a-fA-F-]{16,}$")
EPOCH_SECONDS_RANGE = (946_684_800, 2_208_988_800)   # year 2000 .. 2040


def _name_match(name: str, pattern: str) -> bool:
    return re.search(pattern, name.lower()) is not None


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 12 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_value(value, pii_type: str | None) -> str:
    s = str(value)
    if pii_type is None:
        return s[:40]
    if pii_type in ("card_number", "account_number") and len(s) > 4:
        return "*" * (len(s) - 4) + s[-4:]
    if len(s) <= 2:
        return "*" * len(s)
    return s[0] + "*" * (len(s) - 2) + s[-1]


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    role: str
    confidence: str
    n_unique: int
    unique_ratio: float
    missing_ratio: float
    cardinality: str | None = None
    pii_type: str | None = None
    flags: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    sample_values: list = field(default_factory=list)


@dataclass
class SchemaReport:
    dataset_id: str
    rows: int
    columns: dict
    target: str | None
    target_candidates: list
    detected_at: str
    seconds: float

    def by_role(self) -> dict:
        groups: dict = {}
        for c in self.columns.values():
            groups.setdefault(c.role, []).append(c.name)
        return groups

    def pii_columns(self) -> dict:
        return {c.name: c.pii_type for c in self.columns.values() if c.pii_type}

    def flagged_columns(self) -> dict:
        return {c.name: c.flags for c in self.columns.values() if c.flags}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = {"by_role": self.by_role(), "pii": self.pii_columns(),
                        "flagged": self.flagged_columns()}
        return d

    def save(self, out_dir: str | Path) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"schema_{self.dataset_id}.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        md_path = out_dir / f"schema_{self.dataset_id}.md"
        md_path.write_text(self.to_markdown())
        return json_path, md_path

    def to_markdown(self) -> str:
        lines = [f"# Schema report: {self.dataset_id}", "",
                 f"Rows: {self.rows:,} | Target: `{self.target}` | Detected: {self.detected_at}", "",
                 "| Column | Role | Conf. | dtype | Unique | Missing % | Cardinality | PII | Flags |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for c in self.columns.values():
            lines.append(f"| {c.name} | {c.role} | {c.confidence} | {c.dtype} | {c.n_unique:,} | "
                         f"{c.missing_ratio * 100:.2f} | {c.cardinality or ''} | {c.pii_type or ''} | "
                         f"{'; '.join(c.flags)} |")
        return "\n".join(lines) + "\n"


class SchemaDetector:
    def __init__(self, sample_size: int = 20_000, overrides: dict | None = None,
                 target: str | None = None, low_card_max: int = 20, medium_card_max: int = 200,
                 id_unique_ratio: float = 0.95, parse_success_ratio: float = 0.95, seed: int = 0):
        self.sample_size = sample_size
        self.overrides = overrides or {}
        self.target_override = target
        self.low_card_max = low_card_max
        self.medium_card_max = medium_card_max
        self.id_unique_ratio = id_unique_ratio
        self.parse_ratio = parse_success_ratio
        self.seed = seed

    # ---------- helpers ----------
    def _cardinality(self, n_unique: int) -> str:
        if n_unique <= self.low_card_max:
            return "low"
        return "medium" if n_unique <= self.medium_card_max else "high"

    @staticmethod
    def _numeric_parse_ratio(sample: pd.Series) -> float:
        return float(pd.to_numeric(sample, errors="coerce").notna().mean()) if len(sample) else 0.0

    @staticmethod
    def _datetime_parse_ratio(sample: pd.Series) -> float:
        text = sample.astype(str)
        looks_like_date = text.str.contains(r"[-/:]", regex=True) & ~text.str.fullmatch(r"-?\d+(\.\d+)?")
        if looks_like_date.mean() < 0.5:
            return 0.0
        parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        return float(parsed.notna().mean())

    def _pii_by_name(self, name: str) -> str | None:
        for pii_type, pattern in PII_NAME_HINTS.items():
            if _name_match(name, pattern):
                return pii_type
        return None

    @staticmethod
    def _pii_by_value(sample: pd.Series) -> str | None:
        text = sample.astype(str).head(2000)
        if len(text) == 0:
            return None
        if text.map(lambda v: bool(EMAIL_RE.match(v))).mean() > 0.5:
            return "email"
        if text.map(_luhn_valid).mean() > 0.8:
            return "card_number"
        looks_like_date = text.str.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}.*").mean() > 0.5
        if (not looks_like_date and text.str.contains(r"[()+\s-]").mean() > 0.5
                and text.map(lambda v: bool(PHONE_RE.match(v))).mean() > 0.8):
            return "phone"
        return None

    # ---------- per-column inference ----------
    def _infer_column(self, name: str, s: pd.Series, sample: pd.Series, n_rows: int) -> ColumnSchema:
        non_null = int(s.notna().sum())
        n_unique = int(s.nunique(dropna=True))
        unique_ratio = n_unique / max(non_null, 1)
        col = ColumnSchema(name=name, dtype=str(s.dtype), role="categorical", confidence="medium",
                           n_unique=n_unique, unique_ratio=round(unique_ratio, 6),
                           missing_ratio=round(1 - non_null / max(n_rows, 1), 6))
        ev, flags = col.evidence, col.flags
        is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        is_integer = pd.api.types.is_integer_dtype(s)

        if name in self.overrides:
            col.role, col.confidence = self.overrides[name], "manual"
            ev.append("manual override")
        elif n_unique <= 1:
            col.role, col.confidence = "constant", "high"
            ev.append(f"{n_unique} distinct value(s)")
        elif n_unique == 2:
            col.role, col.confidence = "boolean", "high"
            ev.append("exactly 2 distinct values")
        elif pd.api.types.is_datetime64_any_dtype(s):
            col.role, col.confidence = "datetime", "high"
            ev.append("native datetime dtype")
        elif is_numeric:
            self._infer_numeric(name, s, sample, col, is_integer, n_rows)
        else:
            self._infer_text(name, s, sample, col)

        if col.role in ("categorical", "categorical_code", "entity_id", "boolean"):
            col.cardinality = self._cardinality(n_unique)

        # Sensitive-data detection: values first, then names.
        col.pii_type = (self._pii_by_value(sample)
                        if col.role not in ("numeric", "datetime", "datetime_epoch", "row_index", "categorical_code")
                        else None)
        if col.pii_type:
            ev.append(f"values match {col.pii_type} pattern")
        else:
            col.pii_type = self._pii_by_name(name)
            if col.pii_type:
                ev.append(f"name suggests {col.pii_type}")
        if col.pii_type == "card_number" and col.role not in ("record_id", "entity_id"):
            col.role = "entity_id"
            ev.append("card numbers identify an entity; role set to entity_id")

        # Suspicious patterns worth a human look.
        if col.role in ("categorical", "free_text", "entity_id") and len(sample):
            text = sample.astype(str)
            prefix = _common_prefix(text.tolist())
            at_separator = re.match(r".*[_\-:/|.\s]", prefix)      # only cut at a separator ('fraud_Shop' -> 'fraud_')
            prefix = at_separator.group(0) if at_separator else ""
            if len(prefix) >= 3 and re.search(r"[A-Za-z]", prefix):
                flags.append(f"shared_prefix:'{prefix}'")
                if _name_match(prefix, NAME_HINTS["leak"]):
                    flags.append("prefix_contains_target_related_word")
        if is_numeric and col.role == "numeric" and _name_match(name, NAME_HINTS["geo"]):
            flags.append("geo_coordinate")
        if col.unique_ratio >= self.id_unique_ratio and col.role == "categorical" and n_rows > 100:
            flags.append("near_unique_values")
        if col.cardinality == "high" and col.role == "categorical":
            flags.append("high_cardinality")

        col.sample_values = [mask_value(v, col.pii_type) for v in sample.head(3).tolist()]
        return col

    def _infer_numeric(self, name, s, sample, col, is_integer, n_rows):
        ev = col.evidence
        smin, smax = s.min(), s.max()
        if is_integer and col.n_unique == n_rows and smin in (0, 1) and s.is_monotonic_increasing:
            col.role, col.confidence = "row_index", "high"
            ev.append("unique, monotonic integers starting at 0/1")
        elif (is_integer and smin in (0, 1) and col.unique_ratio >= self.id_unique_ratio
              and re.fullmatch(r"(unnamed: ?\d+|index|row_?(id|num|number|index)?)", name.lower())):
            col.role, col.confidence = "row_index", "medium"
            ev.append("near-unique integers from 0/1 with an export-index name (some values repeated)")
        elif (is_integer and smin in (0, 1)
              and re.fullmatch(r"(unnamed: ?\d+|index|row_?(id|num|number|index)?)", name.lower())
              and s.diff().dropna().eq(1).mean() >= 0.99):
            col.role, col.confidence = "row_index", "medium"
            ev.append("export-index name and counts up by 1 row to row, restarting per file (concatenated exports)")
        elif is_integer and EPOCH_SECONDS_RANGE[0] <= smin and smax <= EPOCH_SECONDS_RANGE[1] and col.unique_ratio > 0.1:
            col.role = "datetime_epoch"
            col.confidence = "high" if _name_match(name, NAME_HINTS["epoch"]) else "medium"
            ev.append("integer values in Unix-seconds range (2000-2040)")
        elif is_integer and len(sample) and sample.astype(str).map(_luhn_valid).mean() > 0.6:
            col.role = "record_id" if col.unique_ratio >= self.id_unique_ratio else "entity_id"
            col.confidence = "high"
            ev.append("integers pass the Luhn card-number checksum")
        elif (is_integer and smin >= 10**11 and col.unique_ratio < self.id_unique_ratio
              and float(s.nunique()) >= 2):
            col.role, col.confidence = "entity_id", "medium"
            ev.append("repeating integer keys with 12+ digits (too long to be a quantity)")
        elif is_integer and _name_match(name, NAME_HINTS["code"]):
            col.role, col.confidence = "categorical_code", "medium"
            ev.append("integer with code-like name; arithmetic on it is meaningless")
        elif is_integer and col.unique_ratio >= self.id_unique_ratio and _name_match(name, NAME_HINTS["id"]):
            col.role, col.confidence = "record_id", "high"
            ev.append("near-unique integers with id-like name")
        elif is_integer and _name_match(name, NAME_HINTS["entity"]) and col.unique_ratio < self.id_unique_ratio:
            col.role, col.confidence = "entity_id", "medium"
            ev.append("repeating integer keys with entity-like name")
        else:
            col.role, col.confidence = "numeric", "high"
            ev.append("numeric dtype")

    def _infer_text(self, name, s, sample, col):
        ev, flags = col.evidence, col.flags
        num_ratio = self._numeric_parse_ratio(sample)
        text_sample = sample.astype(str)
        digit_strings = text_sample.str.fullmatch(r"\d+").mean() if len(text_sample) else 0.0
        leading_zero = text_sample.str.fullmatch(r"0\d+").mean() if len(text_sample) else 0.0
        if num_ratio >= self.parse_ratio and digit_strings >= self.parse_ratio and (
                leading_zero > 0 or _name_match(name, NAME_HINTS["code"])):
            col.role, col.confidence = "categorical_code", "high" if leading_zero > 0 else "medium"
            ev.append("digit strings with leading zeros or a code-like name; not a quantity")
            return
        if num_ratio >= self.parse_ratio:
            col.role, col.confidence = "numeric", "medium"
            flags.append("numeric_stored_as_text")
            ev.append(f"{num_ratio:.1%} of sampled values parse as numbers")
            return
        if 0.5 <= num_ratio < self.parse_ratio:
            flags.append("mixed_types")
            ev.append(f"only {num_ratio:.1%} of values parse as numbers")
        dt_ratio = self._datetime_parse_ratio(sample)
        if dt_ratio >= self.parse_ratio:
            col.role, col.confidence = "datetime", "high"
            flags.append("datetime_stored_as_text")
            ev.append(f"{dt_ratio:.1%} of sampled values parse as datetimes")
            return
        if 0.5 <= dt_ratio < self.parse_ratio:
            flags.append("partially_parseable_datetime")
        text = sample.astype(str)
        avg_len = float(text.str.len().mean()) if len(text) else 0.0
        hex_like = float(text.map(lambda v: bool(HEX_ID_RE.match(v))).mean()) if len(text) else 0.0
        if col.unique_ratio >= self.id_unique_ratio and (hex_like > 0.9 or _name_match(name, NAME_HINTS["id"])):
            col.role, col.confidence = "record_id", "high"
            ev.append("near-unique values that look like identifiers")
        elif col.unique_ratio >= self.id_unique_ratio and avg_len > 30:
            col.role, col.confidence = "free_text", "medium"
            ev.append(f"near-unique long strings (avg length {avg_len:.0f})")
        elif _name_match(name, NAME_HINTS["entity"]) and col.unique_ratio < self.id_unique_ratio:
            col.role, col.confidence = "entity_id", "medium"
            ev.append("repeating keys with entity-like name")
        else:
            col.role, col.confidence = "categorical", "high"
            ev.append(f"text with {col.n_unique:,} distinct values")

    # ---------- dataset-level ----------
    def detect(self, df: pd.DataFrame, dataset_id: str = "dataset") -> SchemaReport:
        t0 = time.time()
        logger.info("START schema detection on %s (%d rows x %d columns)", dataset_id, *df.shape)
        k = min(self.sample_size, len(df))
        sample_idx = df.sample(n=k, random_state=self.seed).index
        columns = {}
        for name in df.columns:
            s = df[name]
            sample = s.loc[sample_idx].dropna()
            columns[str(name)] = self._infer_column(str(name), s, sample, len(df))

        target, candidates = self._choose_target(df, columns)
        self._flag_leakage_names(columns, target)
        self._flag_redundant(df.loc[sample_idx], columns)

        report = SchemaReport(dataset_id=dataset_id, rows=len(df), columns=columns, target=target,
                              target_candidates=candidates,
                              detected_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              seconds=round(time.time() - t0, 2))
        for role, cols in sorted(report.by_role().items()):
            logger.info("%-16s %s", role, cols)
        if report.pii_columns():
            logger.warning("Potentially sensitive columns: %s", report.pii_columns())
        logger.info("Target: %s | done in %.1fs", target, report.seconds)
        return report

    def _choose_target(self, df, columns):
        candidates = []
        for c in columns.values():
            if c.role == "boolean" and _name_match(c.name, NAME_HINTS["target"]):
                positive_rate = float(pd.to_numeric(df[c.name], errors="coerce").mean())
                candidates.append({"column": c.name, "positive_rate": round(positive_rate, 6)})
        if self.target_override:
            target = self.target_override
        elif len(candidates) == 1:
            target = candidates[0]["column"]
        else:
            target = None
            if candidates:
                logger.warning("Multiple target candidates %s; pass target=... explicitly", candidates)
            else:
                logger.warning("No target detected; pass target=... explicitly")
        if target:
            columns[target].role, columns[target].confidence = "target", (
                "manual" if self.target_override else "medium")
            columns[target].evidence.append("binary column with target-like name"
                                            if not self.target_override else "target set manually")
        return target, candidates

    @staticmethod
    def _flag_leakage_names(columns, target):
        for c in columns.values():
            if c.name != target and _name_match(c.name, NAME_HINTS["leak"]):
                c.flags.append("name_suggests_target_related_information")

    @staticmethod
    def _flag_redundant(sample_df, columns, threshold: float = 0.9999):
        series = {}
        for c in columns.values():
            s = sample_df[c.name]
            if c.role in ("numeric", "datetime_epoch"):
                series[c.name] = pd.to_numeric(s, errors="coerce")
            elif c.role == "datetime":
                dt = pd.to_datetime(s, errors="coerce", format="mixed")
                series[c.name] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds()
        if len(series) < 2:
            return
        corr = pd.DataFrame(series).corr().abs()
        names = list(corr.columns)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if corr.loc[a, b] >= threshold:
                    columns[b].flags.append(f"near_duplicate_of:{a}")


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    i = 0
    while i < min(len(lo), len(hi)) and lo[i] == hi[i]:
        i += 1
    return lo[:i]


def detect_schema(df: pd.DataFrame, dataset_id: str = "dataset", **kwargs) -> SchemaReport:
    return SchemaDetector(**kwargs).detect(df, dataset_id)
