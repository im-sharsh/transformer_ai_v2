"""Outlier handling fitted on training rows only.

Transaction data contains legitimate extreme values (large purchases), so the default policy only
FLAGS. Capping and removal are explicit choices; removal applies to training rows only.
Levels: 0 normal, 1 potential_outlier, 2 extreme_outlier.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LEVEL_NAMES = {0: "normal", 1: "potential_outlier", 2: "extreme_outlier"}


@dataclass
class OutlierConfig:
    method: str = "iqr"                 # iqr | robust_z
    potential: float = 1.5              # IQR multiplier (or robust z) for potential outliers
    extreme: float = 3.0                # IQR multiplier (or robust z) for extreme outliers
    log_for_skewed: bool = True         # judge skewed positive columns on log1p scale
    skew_threshold: float = 2.0

    def to_dict(self):
        return asdict(self)


class OutlierDetector:
    def __init__(self, config: OutlierConfig | None = None):
        self.cfg = config or OutlierConfig()
        self.params: dict = {}

    def _scale(self, x: pd.Series, log: bool) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce").astype("float64")
        return np.log1p(x.clip(lower=0)) if log else x

    def fit(self, train: pd.DataFrame, columns: list) -> "OutlierDetector":
        cfg = self.cfg
        for c in columns:
            raw = pd.to_numeric(train[c], errors="coerce").dropna().astype("float64")
            log = bool(cfg.log_for_skewed and raw.min() >= 0 and raw.skew() > cfg.skew_threshold)
            x = self._scale(raw, log)
            if cfg.method == "iqr":
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                lo_p, hi_p = q1 - cfg.potential * iqr, q3 + cfg.potential * iqr
                lo_e, hi_e = q1 - cfg.extreme * iqr, q3 + cfg.extreme * iqr
            else:
                med = x.median()
                mad = 1.4826 * (x - med).abs().median() or x.std() or 1.0
                lo_p, hi_p = med - cfg.potential * mad, med + cfg.potential * mad
                lo_e, hi_e = med - cfg.extreme * mad, med + cfg.extreme * mad
            inv = (lambda v: float(np.expm1(v))) if log else float
            self.params[c] = {"log_scale": log, "potential_low": inv(lo_p), "potential_high": inv(hi_p),
                              "extreme_low": inv(lo_e), "extreme_high": inv(hi_e), "fitted_rows": int(len(raw))}
        return self

    def levels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c, p in self.params.items():
            x = pd.to_numeric(df[c], errors="coerce")
            lvl = np.where((x < p["extreme_low"]) | (x > p["extreme_high"]), 2,
                           np.where((x < p["potential_low"]) | (x > p["potential_high"]), 1, 0))
            out[f"{c}__outlier_level"] = pd.Series(lvl, index=df.index).where(x.notna(), 0).astype("int8")
        out["_outlier_level"] = out.max(axis=1).astype("int8") if len(out.columns) else 0
        return out

    def apply(self, df: pd.DataFrame, policy: str = "flag", is_training: bool = False) -> tuple[pd.DataFrame, dict]:
        """flag: add level columns | cap: clip to the extreme bounds | remove_train: drop extreme rows (training only)."""
        lv = self.levels(df)
        stats = {"rows_in": len(df), "policy": policy,
                 "level_counts": {LEVEL_NAMES[k]: int(v) for k, v in lv["_outlier_level"].value_counts().sort_index().items()}}
        out = df.copy()
        if policy == "flag":
            out = pd.concat([out, lv], axis=1)
        elif policy == "cap":
            changed = {}
            for c, p in self.params.items():
                x = pd.to_numeric(out[c], errors="coerce")
                capped = x.clip(lower=p["extreme_low"], upper=p["extreme_high"])
                changed[c] = int((capped != x).sum() - (x.isna()).sum() * 0)
                out[c] = capped
            stats["values_capped"] = changed
        elif policy == "remove_train":
            if not is_training:
                stats["note"] = "remove_train never removes validation or test rows; rows kept"
            else:
                keep = lv["_outlier_level"] < 2
                out = out[keep]
                stats["rows_removed"] = int((~keep).sum())
        elif policy != "none":
            raise ValueError(policy)
        stats["rows_out"] = len(out)
        return out, stats

    def report(self, df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Rows and target rate per column and level: shows whether 'outliers' are errors or signal."""
        lv = self.levels(df)
        rows = []
        for c in list(self.params) + ["_row"]:
            col = f"{c}__outlier_level" if c != "_row" else "_outlier_level"
            for level, g in df.groupby(lv[col]):
                row = {"column": c if c != "_row" else "<any column>", "level": LEVEL_NAMES[int(level)], "rows": len(g),
                       "share_of_rows": round(len(g) / len(df), 5)}
                if target:
                    y = pd.to_numeric(g[target], errors="coerce")
                    row.update({"target_rate": round(float(y.mean()), 5),
                                "share_of_all_positives": round(float(y.sum() / max(pd.to_numeric(df[target]).sum(), 1)), 4)})
                rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps({"config": self.cfg.to_dict(), "params": self.params}, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OutlierDetector":
        d = json.loads(Path(path).read_text())
        obj = cls(OutlierConfig(**d["config"]))
        obj.params = d["params"]
        return obj
