"""Generic column roles, candidate detection and task detection.

Maps the detailed schema-detector roles onto the framework roles
TARGET, ENTITY, IDENTIFIER, DATETIME, NUMERICAL, CATEGORICAL, TEXT, UNKNOWN,
proposes candidates for the target, entity and datetime columns, and infers the likely task
(binary classification, multiclass classification or regression). Every detection can be overridden.
Nothing here depends on the names of a particular dataset.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src.ingestion.schema_detector import SchemaReport

ROLE_LABELS = ["TARGET", "ENTITY", "IDENTIFIER", "DATETIME", "NUMERICAL", "CATEGORICAL", "TEXT", "UNKNOWN"]
TASKS = ["binary_classification", "multiclass_classification", "regression"]
TASK_LABELS = {"binary_classification": "Binary classification", "multiclass_classification": "Multiclass classification",
               "regression": "Regression"}
DETECTOR_TO_ROLE = {"target": "TARGET", "entity_id": "ENTITY", "record_id": "IDENTIFIER", "row_index": "IDENTIFIER",
                    "datetime": "DATETIME", "datetime_epoch": "DATETIME", "numeric": "NUMERICAL",
                    "categorical": "CATEGORICAL", "categorical_code": "CATEGORICAL", "boolean": "CATEGORICAL",
                    "free_text": "TEXT", "constant": "UNKNOWN"}
TARGET_HINT = (r"(^|_)(target|label|labels|class|y|outcome|churn|churned|default|defaulted|fraud|approved|approval|"
               r"risk|risk_level|response|converted|survived|status|result)($|_)")
ENTITY_HINT = r"(customer|client|user|account|acct|card|(^|_)cc($|_)|member|device|patient|policy|subscriber|borrower)"
LEAK_WORDS = r"(fraud|chargeback|dispute|label|outcome|confirmed|refund|reversal)"


@dataclass
class DatasetRoles:
    columns: dict                       # column -> framework role
    target: str | None
    task: str | None
    entity: str | None
    datetime: str | None
    task_reason: str = ""
    candidates: dict = field(default_factory=dict)
    overridden: dict = field(default_factory=dict)

    def columns_with(self, role: str) -> list:
        return [c for c, r in self.columns.items() if r == role]

    @property
    def numerical(self):
        return self.columns_with("NUMERICAL")

    @property
    def categorical(self):
        return self.columns_with("CATEGORICAL")

    @property
    def identifiers(self):
        return self.columns_with("IDENTIFIER") + self.columns_with("ENTITY")

    def to_frame(self, schema: SchemaReport) -> pd.DataFrame:
        rows = []
        for c, role in self.columns.items():
            cs = schema.columns.get(c)
            rows.append({"column": c, "role": role, "detected_as": cs.role if cs else "",
                         "dtype": cs.dtype if cs else "", "unique": cs.n_unique if cs else None,
                         "missing_pct": round(100 * cs.missing_ratio, 3) if cs else None,
                         "personal_data": cs.pii_type or "" if cs else "",
                         "overridden": c in self.overridden})
        return pd.DataFrame(rows)

    def with_overrides(self, df: pd.DataFrame, role_overrides: dict | None = None, target: str | None = None,
                       task: str | None = None, entity: str | None = None, datetime: str | None = None) -> "DatasetRoles":
        cols = dict(self.columns)
        overridden = dict(self.overridden)
        for c, role in (role_overrides or {}).items():
            if role not in ROLE_LABELS:
                raise ValueError(f"Unknown role '{role}' for column '{c}'")
            if cols.get(c) != role:
                cols[c], overridden[c] = role, f"role {self.columns.get(c)} -> {role}"
        new_target = target if target is not None else self.target
        if new_target and new_target not in df:
            raise ValueError(f"Target column '{new_target}' not in data")
        if new_target != self.target:
            if self.target and cols.get(self.target) == "TARGET":
                cols[self.target] = "UNKNOWN"
            overridden["target"] = f"{self.target} -> {new_target}"
        if new_target:
            cols[new_target] = "TARGET"
        auto_task, reason = detect_task(df[new_target]) if new_target else (None, "no target selected")
        new_task = task or (auto_task if new_target != self.target else self.task or auto_task)
        if task and task != auto_task:
            overridden["task"] = f"detected {auto_task} -> {task}"
            reason = f"set manually (detected: {TASK_LABELS.get(auto_task, auto_task)}; {reason})"
        new_entity = self.entity if entity is None else (entity or None)
        new_datetime = self.datetime if datetime is None else (datetime or None)
        if new_entity:
            cols[new_entity] = "ENTITY"
        if new_datetime:
            cols[new_datetime] = "DATETIME"
        return replace(self, columns=cols, target=new_target, task=new_task, entity=new_entity, datetime=new_datetime,
                       task_reason=reason, overridden=overridden)


def _name(c: str, pattern: str) -> bool:
    return re.search(pattern, str(c).lower()) is not None


def detect_task(target: pd.Series) -> tuple[str, str]:
    s = target.dropna()
    n = int(s.nunique())
    if n <= 1:
        return "binary_classification", f"only {n} distinct value(s): cannot learn; check the target"
    if n == 2:
        return "binary_classification", "exactly 2 distinct values"
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() < 0.95:
        return "multiclass_classification", f"{n} distinct non-numeric labels"
    integer_like = bool(np.all(np.isclose(numeric, np.round(numeric))))
    if integer_like and n <= 20:
        return "multiclass_classification", f"{n} distinct integer values"
    return "regression", f"numeric with {n:,} distinct values"


def target_candidates(df: pd.DataFrame, schema: SchemaReport, limit: int = 5) -> list[dict]:
    last = df.columns[-1]
    out = []
    for c, cs in schema.columns.items():
        if cs.role in ("record_id", "row_index", "entity_id", "datetime", "datetime_epoch", "free_text", "constant") or cs.pii_type:
            continue
        score, why = 0.0, []
        if _name(c, TARGET_HINT):
            score += 3; why.append("name suggests a target")
        if cs.n_unique == 2:
            score += 2; why.append("binary")
            numeric = pd.to_numeric(df[c].dropna(), errors="coerce")
            if numeric.notna().all() and set(numeric.unique()) <= {0, 1}:
                score += 1; why.append("0/1 values")
                rate = float(numeric.mean())
                if 0 < rate < 0.5:
                    score += 0.5; why.append(f"positive rate {rate:.3%}")
        elif 2 < cs.n_unique <= 20 and cs.role in ("categorical", "numeric", "categorical_code"):
            score += 0.5; why.append(f"{cs.n_unique} classes")
        if c == last:
            score += 0.5; why.append("last column")
        if score >= 2 or (score > 0 and _name(c, TARGET_HINT)):
            task, reason = detect_task(df[c])
            out.append({"column": c, "score": score, "evidence": why, "task": task, "task_reason": reason})
    return sorted(out, key=lambda d: -d["score"])[:limit]


def entity_candidates(df: pd.DataFrame, schema: SchemaReport, exclude: tuple = ()) -> list[dict]:
    out = []
    for c, cs in schema.columns.items():
        if c in exclude or cs.role in ("datetime", "datetime_epoch", "constant", "target"):
            continue
        repeating = 1 < cs.n_unique and cs.unique_ratio < 0.5
        if cs.role == "entity_id":
            out.append({"column": c, "evidence": "repeating identifier values", "entities": cs.n_unique})
        elif repeating and _name(c, ENTITY_HINT) and cs.role in ("categorical", "categorical_code", "numeric"):
            out.append({"column": c, "evidence": "entity-like name with repeating values", "entities": cs.n_unique})
    return out


def datetime_candidates(schema: SchemaReport) -> list[dict]:
    out = []
    for c, cs in schema.columns.items():
        if cs.role in ("datetime", "datetime_epoch"):
            kind = "birth date (personal data)" if cs.pii_type == "birth_date" else (
                "numeric epoch seconds" if cs.role == "datetime_epoch" else "parseable dates")
            out.append({"column": c, "evidence": kind, "event_time": cs.pii_type != "birth_date" and cs.role == "datetime"})
    return sorted(out, key=lambda d: not d["event_time"])


def detect_roles(df: pd.DataFrame, schema: SchemaReport, target: str | None = None) -> DatasetRoles:
    cols = {c: DETECTOR_TO_ROLE.get(cs.role, "UNKNOWN") for c, cs in schema.columns.items()}
    t_cands = target_candidates(df, schema)
    chosen = target if target else (schema.target or (t_cands[0]["column"] if t_cands else None))
    for c, r in cols.items():                                   # only one TARGET
        if r == "TARGET" and c != chosen:
            cols[c] = DETECTOR_TO_ROLE.get("boolean") if schema.columns[c].n_unique == 2 else "CATEGORICAL"
    task, reason = detect_task(df[chosen]) if chosen else (None, "no target candidate found; select one")
    if chosen:
        cols[chosen] = "TARGET"
    e_cands = entity_candidates(df, schema, exclude=(chosen,) if chosen else ())
    d_cands = datetime_candidates(schema)
    entity = e_cands[0]["column"] if e_cands else None
    event = next((d["column"] for d in d_cands if d["event_time"]), None)
    return DatasetRoles(columns=cols, target=chosen, task=task, entity=entity, datetime=event, task_reason=reason,
                        candidates={"target": t_cands, "entity": e_cands, "datetime": d_cands})


def target_summary(df: pd.DataFrame, target: str, task: str) -> dict:
    s = df[target]
    if task == "regression":
        x = pd.to_numeric(s, errors="coerce")
        return {"task": task, "missing": int(x.isna().sum()), "mean": float(x.mean()), "median": float(x.median()),
                "std": float(x.std()), "min": float(x.min()), "max": float(x.max()), "skewness": float(x.skew())}
    counts = s.value_counts(dropna=False)
    shares = counts / counts.sum()
    out = {"task": task, "classes": int(s.nunique()), "missing": int(s.isna().sum()),
           "distribution": [{"class": str(k), "rows": int(v), "share": float(shares[k])} for k, v in counts.items()],
           "imbalance_ratio": float(counts.max() / max(counts.min(), 1))}
    if task == "binary_classification" and len(counts) == 2:
        minority = counts.idxmin()
        out.update({"positive_class": str(minority), "positive_rate": float(shares[minority])})
    return out


def entity_summary(df: pd.DataFrame, entity: str) -> dict:
    per = df[entity].value_counts()
    return {"column": entity, "entities": int(per.size), "rows_per_entity_min": int(per.min()),
            "rows_per_entity_median": float(per.median()), "rows_per_entity_max": int(per.max())}


def time_summary(df: pd.DataFrame, column: str, target: str | None = None, task: str | None = None) -> dict:
    from src.profiling.profiler import parse_datetime_column
    t = parse_datetime_column(df[column], "datetime")
    valid = t.dropna()
    monthly = valid.dt.to_period("M").astype(str).value_counts().sort_index()
    out = {"column": column, "min": str(valid.min()) if len(valid) else None, "max": str(valid.max()) if len(valid) else None,
           "invalid": int(t.isna().sum() - df[column].isna().sum()), "missing": int(df[column].isna().sum()),
           "chronological_in_file": bool(valid.is_monotonic_increasing),
           "monthly_rows": {k: int(v) for k, v in monthly.items()}}
    if target and task == "binary_classification":
        y = pd.to_numeric(df[target], errors="coerce")
        if y.dropna().isin([0, 1]).all():
            rate = y[t.notna()].groupby(valid.dt.to_period("M").astype(str)).mean()
            out["monthly_positive_rate"] = {k: float(v) for k, v in rate.items()}
    return out


def leakage_indicators(schema: SchemaReport, roles: DatasetRoles) -> list[dict]:
    """Early, name- and flag-based hints only. Full leakage checks run after feature engineering."""
    out = []
    for c, cs in schema.columns.items():
        if c == roles.target:
            continue
        if _name(c, LEAK_WORDS):
            out.append({"column": c, "indicator": "name refers to an outcome or the target", "severity": "review"})
        if "prefix_contains_target_related_word" in cs.flags:
            out.append({"column": c, "indicator": "values share a prefix containing a target-related word", "severity": "info"})
        for f in cs.flags:
            if f.startswith("near_duplicate_of:"):
                out.append({"column": c, "indicator": f"near-duplicate of {f.split(':', 1)[1]}", "severity": "info"})
        if roles.columns.get(c) in ("IDENTIFIER", "ENTITY"):
            out.append({"column": c, "indicator": "identifier: a model could memorise individual records or entities",
                        "severity": "info"})
    return out


ROLE_TO_DETECTOR = {"NUMERICAL": "numeric", "CATEGORICAL": "categorical", "DATETIME": "datetime", "IDENTIFIER": "record_id",
                    "ENTITY": "entity_id", "TEXT": "free_text", "UNKNOWN": "constant", "TARGET": "target"}


def schema_for_profiling(schema: SchemaReport, roles: DatasetRoles, df: pd.DataFrame) -> SchemaReport:
    """Copy of the schema reflecting user overrides; the profiler's target block is used for 0/1 binary targets only."""
    import copy
    s = copy.deepcopy(schema)
    for c, role in roles.columns.items():
        cs = s.columns.get(c)
        if cs is None:
            continue
        if DETECTOR_TO_ROLE.get(cs.role) != role:
            cs.role = ROLE_TO_DETECTOR[role]
    binary01 = False
    if roles.target and roles.task == "binary_classification":
        y = pd.to_numeric(df[roles.target], errors="coerce").dropna()
        binary01 = len(y) > 0 and set(y.unique()) <= {0, 1}
    s.target = roles.target if binary01 else None
    if roles.target and not binary01 and roles.target in s.columns:
        s.columns[roles.target].role = "categorical" if roles.task != "regression" else "numeric"
    for c, cs in s.columns.items():                             # the profiler reads the entity from the schema
        if cs.role == "entity_id" and c != roles.entity:
            cs.role = "record_id"
    if roles.entity and roles.entity in s.columns:
        s.columns[roles.entity].role = "entity_id"
    return s
