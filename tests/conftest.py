import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger().setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def transactions():
    from src.ingestion.demo_data import make_transactions
    return make_transactions(n_cards=80, days=60, seed=1)


@pytest.fixture(scope="session")
def churn():
    """Structurally different dataset: one row per customer, yes/no target, a signup date."""
    rng = np.random.default_rng(3)
    n = 3000
    spend = rng.gamma(2.0, 30.0, n).round(2)
    plan = rng.choice(["basic", "Basic ", "pro", "enterprise"], n, p=[0.5, 0.05, 0.3, 0.15])
    churn = np.where(rng.random(n) < 0.15 + 0.2 * (spend < 30), "yes", "no")
    return pd.DataFrame({"customer_id": np.arange(100000, 100000 + n),
                         "signup_date": pd.date_range("2021-01-01", periods=n, freq="3h").strftime("%Y-%m-%d"),
                         "monthly_spend": spend, "plan": plan, "support_tickets": rng.poisson(1.2, n),
                         "region": rng.choice(["north", "south", "east", "west"], n), "churn": churn})
