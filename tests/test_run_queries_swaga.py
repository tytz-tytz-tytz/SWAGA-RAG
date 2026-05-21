import re
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / f"{script_name}.py"
    spec = spec_from_file_location(script_name, script_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_ranked_items_filters_invalid_rows():
    mod = load_script_module("run_queries_swaga")
    result = {
        "text_nodes": [
            {"node_id": "n1", "text": "chunk one", "score": 1.0},
            {"node_id": "", "text": "bad id"},
            {"node_id": "n2", "text": ""},
            {"node_id": "n3", "text": "chunk three"},
            "not a dict",
        ]
    }
    items = mod.extract_ranked_items(result)
    assert items == [
        {"chunk_id": "n1", "text": "chunk one", "score": 1.0},
        {"chunk_id": "n3", "text": "chunk three"},
    ]


def test_sanitize_run_name():
    mod = load_script_module("run_queries_swaga")
    assert mod.sanitize_run_name("stable baseline/1") == "stable_baseline_1"
    assert mod.sanitize_run_name("...") == "run"


def test_compute_run_id_cli_override():
    mod = load_script_module("run_queries_swaga")
    cfg = {"run": {"name": "stable_baseline", "append_timestamp": True}}
    assert mod.compute_run_id(cfg, "my run") == "my_run"


def test_compute_run_id_with_timestamp_suffix():
    mod = load_script_module("run_queries_swaga")
    cfg = {"run": {"name": "stable_baseline", "append_timestamp": True}}
    run_id = mod.compute_run_id(cfg, None)
    assert re.match(r"^stable_baseline__\d{8}_\d{6}$", run_id)


def test_get_out_dir_from_config():
    mod = load_script_module("run_queries_swaga")
    cfg = {"run": {"out_dir": "artifacts/swaga_rag_results"}}
    assert mod.get_out_dir_from_config(cfg).as_posix() == "artifacts/swaga_rag_results"


def test_get_out_dir_from_config_raises_when_missing():
    mod = load_script_module("run_queries_swaga")
    with pytest.raises(ValueError):
        mod.get_out_dir_from_config({})
