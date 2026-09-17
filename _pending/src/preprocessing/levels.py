"""Data preparation levels.

E0  Raw                 minimal formatting: the same identifier / personal-data exclusion policy as every level
E1  Quality processed   type normalisation, validation and cleaning, categorical normalisation, rare-category
                        handling, numeric transformations, datetime processing (parsing, age from birth date)
E2  Feature engineered  E1 + hour / day of week / month / weekend + point-in-time history features

All levels share the same split and the same test rows. Everything statistical is fitted on training rows only,
and every history feature uses only transactions strictly earlier in time.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.behavioral_features import HistoryConfig, card_history, merchant_history
from src.ingestion.schema_detector import SchemaReport
from src.preprocessing.cleaner import CleaningConfig, clean_dataset
from src.preprocessing.datetime_features import add_time_features
from src.preprocessing.pipeline import ColumnRoles, FittedPreprocessor, PreprocessingConfig
from src.preprocessing.sampling import SampleConfig, draw_sample, sample_report
from src.profiling.profiler import parse_datetime_column
from src.quality.leakage_detector import LeakageDetector, check_point_in_time
from src.quality.quality_engine import assess_quality
from src.representation.transaction_formatter import find_quasi_identifiers

logger = logging.getLogger("LEVELS")

LEVELS = {"E0": "Raw", "E1": "Quality processed", "E2": "Feature engineered"}
META = ["_row_id", "_split", "_target", "_weight", "_new_entity"]
PII_EXCLUDE = {"person_name", "address", "card_number", "account_number", "email", "phone", "government_id",
               "location", "birth_date"}
HINTS = {"amount": r"(^|_)(amt|amount|value|price|total|sum)($|_)", "category": r"(categ|mcc|(^|_)type($|_)|segment)",
         "merchant": r"(merchant|payee|store|vendor|shop)"}
HISTORY_FEATURES = ["card_seconds_since_prev", "card_count_1h", "card_count_24h", "card_count_7d", "card_count_30d",
                    "card_amount_sum_24h", "card_amount_mean_before", "card_amount_std_before",
                    "card_amount_ratio_to_mean", "card_amount_zscore", "card_first_time_category",
                    "card_category_share_before", "card_first_time_merchant", "merchant_count_24h",
                    "merchant_count_7d", "merchant_amount_mean_before", "merchant_amount_ratio_to_mean"]
TIME_FEATURES = ["hour", "day_of_week", "month", "is_weekend"]


@dataclass
class Roles:
    target: str
    time: str | None
    entity: str | None
    amount: str | None
    category: str | None
    merchant: str | None
    birth: list
    excluded: dict                       # column -> reason; identical for every level

    def as_dict(self):
        return self.__dict__.copy()


def _match(name: str, pattern: str) -> bool:
    return re.search(pattern, name.lower()) is not None


def infer_roles(df: pd.DataFrame, schema: SchemaReport, hints: dict | None = None, quasi_sample_rows: int = 200_000) -> Roles:
    hints = {k: v for k, v in (hints or {}).items() if v}
    by_role = schema.by_role()
    cols = schema.columns
    target = hints.get("target") or schema.target
    if not target or target not in df:
        raise ValueError("No binary target column found. Choose the target column explicitly.")
    datetimes = [c for c in by_role.get("datetime", []) if cols[c].pii_type != "birth_date"]
    time_col = hints.get("time_column") or (datetimes[0] if datetimes else None)
    entity = hints.get("entity_column") or (by_role.get("entity_id") or [None])[0]
    numeric = [c for c in by_role.get("numeric", []) if c != target]
    categorical = [c for c in by_role.get("categorical", []) if not cols[c].pii_type]
    pick = lambda key, pool: hints.get(f"{key}_column") or next((c for c in pool if _match(c, HINTS[key])), None)
    roles = Roles(target=target, time=time_col, entity=entity, amount=pick("amount", numeric),
                  category=pick("category", categorical), merchant=pick("merchant", categorical),
                  birth=[c for c in by_role.get("datetime", []) if cols[c].pii_type == "birth_date"], excluded={})

    for role in ["record_id", "row_index", "entity_id", "constant"]:
        for c in by_role.get(role, []):
            roles.excluded[c] = f"{role.replace('_', ' ')}: carries no generalisable signal"
    for c, cs in cols.items():
        if cs.pii_type in PII_EXCLUDE and c not in roles.excluded:
            roles.excluded[c] = f"personal data ({cs.pii_type})"
    if entity and entity in df:
        candidates = [c for c in df.columns if c not in roles.excluded and c not in (target, time_col, entity)
                      and cols.get(c) is not None and cols[c].role not in ("datetime", "datetime_epoch", "free_text")]
        frame = df.sample(min(len(df), quasi_sample_rows), random_state=0) if len(df) > quasi_sample_rows else df
        for c, reason in find_quasi_identifiers(frame, entity, candidates).items():
            if c not in (roles.amount, roles.category, roles.merchant):
                roles.excluded[c] = reason
    roles.excluded.pop(target, None)
    return roles


@dataclass
class PreparedLevel:
    level: str
    frames: dict                        # split -> DataFrame (META + features)
    numeric: list
    categorical: list
    info: dict = field(default_factory=dict)

    @property
    def features(self):
        return self.numeric + self.categorical


class DataPreparer:
    def __init__(self, df: pd.DataFrame, schema: SchemaReport, roles: Roles, config: dict, rows: int | str = 20000,
                 seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.df.insert(0, "_row_id", [f"r{i}" for i in range(len(self.df))]) if "_row_id" not in self.df else None
        self.schema, self.roles, self.cfg, self.rows, self.seed = schema, roles, config, rows, seed
        self.sample_ids: pd.DataFrame | None = None
        self.split_info: dict = {}
        self._history: pd.DataFrame | None = None
        self.cache: dict = {}

    # ------------------------------------------------------------------ target, split, sample
    def _binary_target(self) -> pd.Series:
        y = self.df[self.roles.target]
        num = pd.to_numeric(y, errors="coerce")
        if num.notna().mean() > 0.99 and set(num.dropna().unique()) <= {0, 1}:
            return num
        values = y.dropna().astype(str).value_counts()
        if len(values) != 2:
            raise ValueError(f"Target '{self.roles.target}' must have exactly two values; found {len(values)}")
        positive = values.index[-1]                                  # minority class is the positive class
        return (y.astype(str) == positive).astype(float).where(y.notna())

    def prepare_split(self) -> dict:
        t0 = time.time()
        cfg, df = self.cfg, self.df
        f_train, f_val, _ = cfg["split"]["fractions"]
        target = self._binary_target()
        split = pd.Series("train", index=df.index, dtype="object")
        boundaries = {}
        if cfg["split"]["method"] == "temporal" and self.roles.time:
            t = parse_datetime_column(df[self.roles.time], "datetime")
            q1, q2 = t.quantile(f_train), t.quantile(f_train + f_val)
            split[t >= q1] = "validation"
            split[t >= q2] = "test"
            split[t.isna()] = "unassigned"
            boundaries = {"train": f"{t.min()} to before {q1}", "validation": f"{q1} to before {q2}",
                          "test": f"{q2} to {t.max()}", "method": "temporal (quantiles of event time)"}
        else:
            rng = np.random.default_rng(self.seed).random(len(df))
            split[rng >= f_train] = "validation"
            split[rng >= f_train + f_val] = "test"
            boundaries = {"method": "random (no usable time column)" if cfg["split"]["method"] == "temporal" else "random"}
        split[target.isna()] = "unassigned"
        frame = pd.DataFrame({"_row_id": df["_row_id"], "split": split, "y": target})
        frame = frame[frame["split"] != "unassigned"]

        s = cfg["sampling"]
        if self.rows == "full" or int(self.rows) >= len(frame):
            sizes = frame["split"].value_counts().to_dict()
            scfg = SampleConfig(sizes={k: sizes.get(k, 0) for k in ["train", "validation", "test"]}, positive_share={},
                                include_all_positives={}, seed=self.seed)
        else:
            n = int(self.rows)
            sizes = {"train": int(n * f_train), "validation": int(n * f_val), "test": n - int(n * f_train) - int(n * f_val)}
            scfg = SampleConfig(sizes=sizes, positive_share={"train": s["train_positive_share"],
                                                             "validation": s["validation_positive_share"]},
                                include_all_positives={"test": s["keep_all_test_positives"]},
                                max_positive_share=s["max_test_positive_share"], seed=self.seed)
        sample = draw_sample(frame, "y", scfg)
        if self.roles.entity:
            train_entities = set(df.loc[split == "train", self.roles.entity])
            entity_of = df.set_index("_row_id")[self.roles.entity]
            sample["_new_entity"] = ~sample["_row_id"].map(entity_of).isin(train_entities)
        else:
            sample["_new_entity"] = False
        self.sample_ids = sample.rename(columns={"split": "_split", "y": "_target"})
        self.split_info = {"boundaries": boundaries, "rows_available": int(len(df)),
                           "rows_unassigned": int((split == "unassigned").sum()),
                           "mode_rows": self.rows, "sample": sample_report(sample.rename(columns={"y": "is_y"}), "is_y"),
                           "seconds": round(time.time() - t0, 2)}
        return self.split_info

    def _sample_rows(self) -> pd.DataFrame:
        if self.sample_ids is None:
            self.prepare_split()
        return self.sample_ids.merge(self.df, on="_row_id", how="left")

    # ------------------------------------------------------------------ levels
    def build(self, level: str) -> PreparedLevel:
        if level in self.cache:
            return self.cache[level]
        t0 = time.time()
        builder = {"E0": self._build_e0, "E1": self._build_e1, "E2": self._build_e2}[level]
        prepared = builder()
        prepared.info.update({"level": level, "name": LEVELS[level], "seconds": round(time.time() - t0, 2),
                              "features": len(prepared.features),
                              "rows": {s: int(len(f)) for s, f in prepared.frames.items()},
                              "positives": {s: int(f["_target"].sum()) for s, f in prepared.frames.items()},
                              "split": self.split_info})
        self.cache[level] = prepared
        return prepared

    def _finish(self, level, frame, features, steps, extra=None) -> PreparedLevel:
        features = [c for c in features if c in frame]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]
        categorical = [c for c in features if c not in numeric]
        frames = {s: frame[frame["_split"] == s][META + features].reset_index(drop=True) for s in ["train", "validation", "test"]}
        return PreparedLevel(level, frames, numeric, categorical, info={"steps": steps, **(extra or {})})

    def _build_e0(self) -> PreparedLevel:
        rows = self._sample_rows()
        r = self.roles
        features = [c for c in self.df.columns if c not in META and c not in r.excluded and c != r.target]
        steps = ["Minimal formatting: values kept exactly as loaded",
                 f"Excluded {len(r.excluded)} identifier / personal-data / quasi-identifier columns (same policy for every level)"]
        return self._finish("E0", rows, features, steps, {"excluded": r.excluded})

    def _e1_frame(self) -> tuple[pd.DataFrame, list, list, dict]:
        r, cfg = self.roles, self.cfg
        rows = self._sample_rows()
        content_cols = [c for c in self.df.columns if c != "_row_id"]
        content = rows[content_cols]
        quality = assess_quality(content, self.schema)
        protected = rows["_split"] != "train"
        cleaning = clean_dataset(rows[["_row_id"] + content_cols].assign(_split=rows["_split"]), self.schema, quality,
                                 protected=protected, config=CleaningConfig())
        cleaned = cleaning.df.merge(rows[["_row_id", "_target", "_weight", "_new_entity"]], on="_row_id", how="left")
        steps = [f"Quality score on sampled rows: {quality.scores['overall']:.1f}",
                 f"Cleaning: {cleaning.summary['rows_in']:,} rows in, {cleaning.summary['rows_quarantined']:,} quarantined, "
                 f"{cleaning.summary['duplicates_removed']:,} duplicates removed (validation/test rows are never removed)",
                 f"Dropped columns: {cleaning.summary['dropped_columns'] or 'none'}"]
        added = []
        if r.time and r.birth and r.time in cleaned:
            with_age, cols = add_time_features(cleaned[[r.time] + [b for b in r.birth if b in cleaned]], r.time,
                                               [b for b in r.birth if b in cleaned])
            for c in cols:
                if c.startswith("age_years"):
                    cleaned[c] = with_age[c].astype("float64")
                    added.append(c)
            steps.append(f"Datetime processing: parsed timestamps; derived {added} from birth date (raw birth date stays excluded)")
        by_role = self.schema.by_role()
        numeric = [c for c in by_role.get("numeric", []) if c in cleaned and c not in r.excluded and c != r.target] + added
        categorical = [c for c in by_role.get("categorical", []) + by_role.get("categorical_code", []) + by_role.get("boolean", [])
                       if c in cleaned and c not in r.excluded and c != r.target]
        pre = FittedPreprocessor(PreprocessingConfig(time_features=False, rare_min_frequency=cfg["preprocessing"]["rare_min_frequency"],
                                                     n_bins=cfg["preprocessing"]["n_bins"],
                                                     log_skew_threshold=cfg["preprocessing"]["log_skew_threshold"]))
        pre.fit(cleaned[cleaned["_split"] == "train"], ColumnRoles(numeric=numeric, categorical=categorical, event_time=None))
        transformed, stats = pre.transform(cleaned)
        indicators = [c for c in transformed.columns if c.endswith("__was_missing")]
        bins = [f"{c}__bin" for c in numeric if f"{c}__bin" in transformed]
        steps += ["Fitted on training rows only: missing-value handling, rare-category grouping, numeric transforms "
                  f"(log for skewed: {[c for c, p in pre.numeric.params.items() if p['log']]}, robust scaling, decile bins)",
                  "Absolute timestamps removed from model inputs: later periods lie outside the training range"]
        audit = {"cleaning": cleaning.summary, "audit_log": cleaning.audit.entries, "preprocessor": pre.to_dict(),
                 "transform_stats": stats, "quality_scores": quality.scores}
        return transformed, numeric + bins + indicators, categorical, {"steps": steps, "audit": audit}

    # Only structural leaks are removed automatically. Strong predictive power alone (e.g. the amount) is flagged
    # for human review, never removed: a legitimate signal can be very predictive.
    AUTO_REMOVE_CHECKS = {"post_event_time", "target_word_in_text"}

    def _leakage(self, frame, features, level) -> tuple[list, list]:
        """Run the leakage detector; remove only high-risk structural leaks, with the reason logged."""
        if not self.roles.time:
            return [], []
        check = frame.assign(_event_time=self._event_times(frame), _y=frame["_target"])
        cutoff = check.loc[check["_split"] != "train", "_event_time"].min()
        det = LeakageDetector(target="_y", event_time="_event_time", entity=None, cutoff=str(cutoff))
        rep = det.detect(check, dataset_id=level, features=[f for f in features if f in check])
        findings = [f.__dict__ for f in rep.findings]
        removed = sorted({f["feature"] for f in findings
                          if f["leakage_risk"] == "high" and f["check"] in self.AUTO_REMOVE_CHECKS})
        return findings, removed

    def _event_times(self, frame):
        times = parse_datetime_column(self.df.set_index("_row_id")[self.roles.time], "datetime")
        return frame["_row_id"].map(times)

    def _build_e1(self) -> PreparedLevel:
        frame, numeric, categorical, extra = self._e1_frame()
        findings, removed = self._leakage(frame, numeric + categorical, "E1")
        extra["steps"].append(f"Leakage checks: {len(findings)} findings; removed structural leaks: {removed or 'none'} (other findings are for review)")
        extra["leakage"] = findings
        return self._finish("E1", frame, [c for c in numeric + categorical if c not in removed], extra["steps"], extra)

    def _history_features(self) -> pd.DataFrame:
        if self._history is not None:
            return self._history
        r = self.roles
        if not (r.entity and r.time and r.amount):
            raise ValueError("E2 needs entity, time and amount columns (set them in Processing or config.yaml)")
        base = pd.DataFrame({"_row_id": self.df["_row_id"], "entity": self.df[r.entity],
                             "time": parse_datetime_column(self.df[r.time], "datetime"),
                             "amount": pd.to_numeric(self.df[r.amount], errors="coerce")})
        if r.category:
            base["category"] = self.df[r.category].astype(str).str.strip()
        if r.merchant:
            base["merchant"] = self.df[r.merchant].astype(str).str.strip()
        base = base.dropna(subset=["time"])
        hc = HistoryConfig(entity="entity", time="time", amount="amount", category="category" if r.category else None,
                           merchant="merchant" if r.merchant else None, windows=tuple(self.cfg["features"]["windows"]))
        parts = [card_history(base, hc)]
        if r.merchant:
            parts.append(merchant_history(base, hc))
        hist = pd.concat(parts, axis=1)
        hist.insert(0, "_row_id", base["_row_id"].values)
        # point-in-time verification on a few entities
        ents = base["entity"].drop_duplicates().sample(min(10, base["entity"].nunique()), random_state=self.seed)
        sub = base[base["entity"].isin(ents)]
        self.pit = check_point_in_time(sub, lambda f: card_history(f, hc), "time", "entity", n_samples=min(100, len(sub)))
        self._history = hist
        return hist

    def _build_e2(self) -> PreparedLevel:
        frame, numeric, categorical, extra = self._e1_frame()
        r = self.roles
        t = self._event_times(frame)
        for name, values in {"hour": t.dt.hour, "day_of_week": t.dt.dayofweek, "month": t.dt.month,
                             "is_weekend": (t.dt.dayofweek >= 5).astype("float64")}.items():
            frame[name] = values.astype("float64")
        hist = self._history_features()
        keep = [c for c in HISTORY_FEATURES if c in hist]
        frame = frame.merge(hist[["_row_id"] + keep], on="_row_id", how="left")
        extra["steps"] += [f"Time features: {TIME_FEATURES}",
                           f"History features from strictly earlier transactions of the same {r.entity} "
                           f"(no past labels): {len(keep)} features",
                           f"Point-in-time check on real rows: {'passed' if self.pit['passed'] else 'FAILED'} "
                           f"({self.pit['rows_checked']} rows)"]
        feats = numeric + TIME_FEATURES + keep + categorical
        findings, removed = self._leakage(frame, feats, "E2")
        extra["steps"].append(f"Leakage checks: {len(findings)} findings; removed structural leaks: {removed or 'none'} "
                              "(other findings are for review)")
        extra["leakage"], extra["point_in_time"] = findings, self.pit
        return self._finish("E2", frame, [c for c in feats if c not in removed], extra["steps"], extra)
