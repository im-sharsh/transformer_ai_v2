from src.utils.config import deep_merge, load_config
from src.utils.hardware import detect_hardware, recommendations


def test_config_loads_and_merges():
    cfg = load_config(overrides={"project": {"seed": 7}})
    assert cfg["project"]["seed"] == 7 and "profiling" in cfg
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_hardware_detection():
    hw = detect_hardware()
    assert {"cuda", "gpu_name", "cpu_count", "device"} <= set(hw)
    assert hw["device"] in ("cpu", "cuda")
    recs = recommendations(hw)
    assert recs and all(level in ("ok", "warn") for level, _ in recs)
