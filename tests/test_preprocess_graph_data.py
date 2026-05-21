from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / f"{script_name}.py"
    spec = spec_from_file_location(script_name, script_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_ws():
    mod = load_script_module("preprocess_graph_data")
    assert mod.normalize_ws(" a\t\tb \r\n  c ") == "a b \nc"


def test_strip_leading_page_number():
    mod = load_script_module("preprocess_graph_data")
    assert mod.strip_leading_page_number("12 Hello world") == "Hello world"
    assert mod.strip_leading_page_number("12 345") == "12 345"


def test_alpha_ratio():
    mod = load_script_module("preprocess_graph_data")
    assert mod.alpha_ratio("abc123") == 0.5
    assert mod.alpha_ratio("   ") == 0.0


def test_is_noise_text():
    mod = load_script_module("preprocess_graph_data")
    assert mod.is_noise_text(None) is True
    assert mod.is_noise_text(" ") is True
    assert mod.is_noise_text("12") is True
    assert mod.is_noise_text(".") is True
    assert mod.is_noise_text("Good content sentence.") is False
