from src.ingestion.roles import detect_roles
from src.ingestion.schema_detector import detect_schema
from src.profiling.profiler import profile_dataset


def test_profile_transactions(transactions):
    s = detect_schema(transactions, "tx")
    p = profile_dataset(transactions, s)
    assert p.overview["rows"] == len(transactions) and p.overview["missing_cells"] == 0
    assert p.overview["exact_duplicate_rows"] == 0
    assert p.columns["amt"].stats["skewness"] > 1
    assert p.columns["merchant"].stats["rare_category_count"] >= 0
    assert "<masked" in next(iter(p.columns["first"].stats["top_values"]))   # personal data masked


def test_profile_generic_dataset(churn):
    s = detect_schema(churn, "churn")
    r = detect_roles(churn, s)
    p = profile_dataset(churn, s)
    assert p.overview["rows"] == len(churn)
    assert p.columns["plan"].stats["case_or_whitespace_variants"] >= 1      # 'basic' vs 'Basic '
    assert "monthly_spend" in p.columns and r.task == "binary_classification"
