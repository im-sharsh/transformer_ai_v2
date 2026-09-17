"""Synthetic transaction data shaped like the demo dataset (for tests and offline demos only).

Fraud is planted as short night-time bursts of unusually large purchases on a card, so that
time features and history features carry real, learnable signal. Results on this data say
nothing about real-world performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["grocery_pos", "gas_transport", "home", "shopping_pos", "kids_pets", "shopping_net", "entertainment",
              "food_dining", "personal_care", "health_fitness", "misc_pos", "misc_net", "grocery_net", "travel"]
STATES = ["NY", "CA", "TX", "FL", "PA", "OH", "IL", "MI", "AL", "MO"]


def _luhn_complete(prefix: str, rng) -> int:
    body = prefix + "".join(str(d) for d in rng.integers(0, 10, 15 - len(prefix)))
    total = 0
    for i, d in enumerate(int(c) for c in reversed(body)):
        total += d if i % 2 else (d * 2 - 9 if d * 2 > 9 else d * 2)
    return int(body + str((10 - total % 10) % 10))


def make_transactions(n_cards: int = 150, days: int = 120, per_card_per_day: float = 1.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-01-01")
    rows = []
    cards = []
    for c in range(n_cards):
        cards.append({"cc_num": _luhn_complete("4", rng), "first": f"Name{c % 40}", "last": f"Family{c}",
                      "gender": rng.choice(["F", "M"]), "street": f"{c} Main Street", "city": f"City{c}",
                      "state": rng.choice(STATES), "zip": int(rng.integers(1000, 99999)), "lat": rng.uniform(25, 48),
                      "long": rng.uniform(-120, -70), "city_pop": int(rng.integers(500, 2_000_000)),
                      "job": f"Job{c}", "dob": (start - pd.Timedelta(days=int(rng.integers(18 * 365, 80 * 365)))).date().isoformat(),
                      "base_amt": rng.uniform(20, 90)})
    fraud_cards = rng.choice(n_cards, max(3, n_cards // 6), replace=False)
    for ci, card in enumerate(cards):
        n = rng.poisson(per_card_per_day * days)
        ts = start + pd.to_timedelta(np.sort(rng.integers(8 * 3600, days * 86400, n)), unit="s")
        for t in ts:
            big = rng.random() < 0.03                               # occasional legitimate large purchase
            amt = rng.uniform(250, 1500) if big else rng.lognormal(np.log(card["base_amt"]), 0.8)
            rows.append((card, t, float(np.round(amt, 2)), rng.choice(CATEGORIES[:11]), 0))
        if ci in fraud_cards:
            burst_start = start + pd.Timedelta(days=int(rng.integers(5, days - 2)), hours=int(rng.integers(0, 3)))
            for k in range(int(rng.integers(4, 9))):
                t = burst_start + pd.Timedelta(minutes=int(k * rng.integers(5, 40)))
                rows.append((card, t, float(np.round(rng.uniform(60, 1200), 2)),
                             rng.choice(["shopping_net", "misc_net", "grocery_pos"]), 1))
    df = pd.DataFrame(rows, columns=["card", "trans_date_trans_time", "amt", "category", "is_fraud"])
    df = df.sort_values("trans_date_trans_time", kind="mergesort").reset_index(drop=True)
    cols = ["cc_num", "first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob"]
    info = pd.DataFrame([{k: c[k] for k in cols} for c in df["card"]])
    out = pd.concat([info, df.drop(columns=["card"])], axis=1)
    out.insert(0, "Unnamed: 0", np.arange(len(out)))
    out["merchant"] = "fraud_Shop" + pd.Series(rng.integers(0, 60, len(out))).astype(str) + " Ltd"
    out["trans_num"] = [f"{x:032x}" for x in rng.integers(0, 2**62, len(out))]
    ts = pd.to_datetime(out["trans_date_trans_time"])
    out["unix_time"] = ((ts - pd.Timestamp("1970-01-01")).dt.total_seconds() - 220_924_800).astype("int64")
    out["merch_lat"] = out["lat"] + rng.uniform(-1, 1, len(out))
    out["merch_long"] = out["long"] + rng.uniform(-1, 1, len(out))
    out["trans_date_trans_time"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    order = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last", "gender",
             "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "trans_num", "unix_time",
             "merch_lat", "merch_long", "is_fraud"]
    return out[order]
