import json

import numpy as np
import pandas as pd
import pytest

from src.ingestion.loader import DatasetIntegrityError, UnsupportedFormatError, load_dataset


def _frame():
    return pd.DataFrame({"id": [1, 2, 3], "amount": [10.5, np.nan, 3.0], "when": ["2024-01-01", "2024-01-02", "bad"]})


@pytest.mark.parametrize("fmt", ["csv", "json", "jsonl"])
def test_load_formats(tmp_path, fmt):
    df = _frame()
    path = tmp_path / f"data.{fmt}"
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "json":
        df.to_json(path, orient="records")
    else:
        df.to_json(path, orient="records", lines=True)
    before = path.read_bytes()
    ds = load_dataset(path, metadata_dir=tmp_path / "meta")
    assert ds.df.shape == (3, 3) and ds.metadata.missing_cells == 1
    assert path.read_bytes() == before                          # raw file never modified
    assert json.loads((tmp_path / "meta" / f"{ds.metadata.dataset_id}.json").read_text())["rows"] == 3


def test_rejects_bad_input(tmp_path):
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(UnsupportedFormatError):
        load_dataset(tmp_path / "x.txt", metadata_dir=tmp_path)
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(DatasetIntegrityError):
        load_dataset(tmp_path / "empty.csv", metadata_dir=tmp_path)
