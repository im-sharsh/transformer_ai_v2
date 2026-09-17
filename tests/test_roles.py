import numpy as np
import pandas as pd

from src.ingestion.roles import (TASKS, detect_roles, detect_task, entity_summary, leakage_indicators, target_summary,
                                 time_summary)
from src.ingestion.schema_detector import detect_schema


def test_transaction_roles(transactions):
    s = detect_schema(transactions, "tx")
    r = detect_roles(transactions, s)
    assert (r.target, r.task, r.entity, r.datetime) == ("is_fraud", "binary_classification", "cc_num", "trans_date_trans_time")
    assert set(r.identifiers) >= {"trans_num", "Unnamed: 0", "cc_num"}
    assert "amt" in r.numerical and "category" in r.categorical
    summ = target_summary(transactions, r.target, r.task)
    assert 0 < summ["positive_rate"] < 0.05 and summ["positive_class"] == "1"


def test_no_dependence_on_column_names(transactions):
    renamed = transactions.rename(columns={c: f"col_{i}" for i, c in enumerate(transactions.columns)})
    s = detect_schema(renamed, "renamed")
    r = detect_roles(renamed, s)
    original = detect_roles(transactions, detect_schema(transactions, "tx"))
    mapping = dict(zip(transactions.columns, renamed.columns))
    assert r.target == mapping["is_fraud"]                       # binary 0/1, imbalanced, last column
    assert r.datetime == mapping["trans_date_trans_time"]
    assert r.entity == mapping["cc_num"]                         # found from values (card checksum), not the name
    assert r.columns[mapping["trans_num"]] == "IDENTIFIER" and r.columns[mapping["amt"]] == "NUMERICAL"
    assert original.task == r.task


def test_churn_dataset(churn):
    s = detect_schema(churn, "churn")
    r = detect_roles(churn, s)
    assert r.target == "churn" and r.task == "binary_classification"
    assert r.columns["customer_id"] == "IDENTIFIER"              # unique per row: an identifier, not an entity
    assert r.entity is None and r.datetime == "signup_date"
    assert "monthly_spend" in r.numerical and "plan" in r.categorical
    summ = target_summary(churn, "churn", r.task)
    assert summ["positive_class"] == "yes" and summ["classes"] == 2


def test_task_detection():
    assert detect_task(pd.Series([0, 1, 1, 0]))[0] == "binary_classification"
    assert detect_task(pd.Series(["low", "medium", "high"] * 10))[0] == "multiclass_classification"
    assert detect_task(pd.Series([1, 2, 3, 4, 5] * 10))[0] == "multiclass_classification"
    assert detect_task(pd.Series(np.random.default_rng(0).normal(100, 20, 500)))[0] == "regression"


def test_overrides(churn):
    s = detect_schema(churn, "churn")
    r = detect_roles(churn, s)
    r2 = r.with_overrides(churn, role_overrides={"support_tickets": "CATEGORICAL"}, target="monthly_spend")
    assert r2.target == "monthly_spend" and r2.task == "regression" and r2.columns["churn"] != "TARGET"
    assert r2.columns["support_tickets"] == "CATEGORICAL" and "support_tickets" in r2.overridden
    r3 = r2.with_overrides(churn, task="multiclass_classification")
    assert r3.task == "multiclass_classification" and "task" in r3.overridden
    assert set(TASKS) >= {r.task, r2.task, r3.task}


def test_summaries(transactions):
    s = detect_schema(transactions, "tx")
    r = detect_roles(transactions, s)
    t = time_summary(transactions, r.datetime, r.target, r.task)
    assert t["invalid"] == 0 and t["min"] < t["max"] and sum(t["monthly_rows"].values()) == len(transactions)
    e = entity_summary(transactions, r.entity)
    assert e["entities"] == transactions["cc_num"].nunique()
    ind = leakage_indicators(s, r)
    assert any(i["column"] == "merchant" for i in ind)              # 'fraud_' prefix in every merchant name


def test_schema_for_profiling_reflects_overrides(churn):
    from src.ingestion.roles import schema_for_profiling
    from src.profiling.profiler import profile_dataset
    s = detect_schema(churn, "churn")
    r = detect_roles(churn, s).with_overrides(churn, role_overrides={"support_tickets": "CATEGORICAL"})
    ps = schema_for_profiling(s, r, churn)
    assert ps.columns["support_tickets"].role == "categorical" and ps.target is None   # yes/no target: not a 0/1 block
    p = profile_dataset(churn, ps)
    assert "top_values" in p.columns["support_tickets"].stats and s.columns["support_tickets"].role != "categorical"
