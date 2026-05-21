import json
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


def test_filter_only():
    mod = load_script_module("run_param_experiments")
    cfgs = [Path("a.json"), Path("b.json"), Path("c.json")]
    out = mod.filter_only(cfgs, "a,c")
    assert [p.stem for p in out] == ["a", "c"]


def test_build_resolved_config_sets_required_run_fields(tmp_path: Path):
    mod = load_script_module("run_param_experiments")
    src_cfg = tmp_path / "stable.json"
    src_cfg.write_text(json.dumps({"score": {"mode": "full"}}), encoding="utf-8")

    out_dir = tmp_path / "out" / "stable"
    resolved = mod.build_resolved_config(src_cfg, out_dir)

    assert resolved["run"]["out_dir"] == out_dir.as_posix()
    assert resolved["run"]["append_timestamp"] is False
    assert resolved["run"]["name"] == "stable"
    assert resolved["score"]["mode"] == "full"


def test_prepare_out_dir_overwrite(tmp_path: Path):
    mod = load_script_module("run_param_experiments")
    out_dir = tmp_path / "exp"
    out_dir.mkdir(parents=True)
    (out_dir / "old.txt").write_text("x", encoding="utf-8")

    mod.prepare_out_dir(out_dir, overwrite=True)
    assert out_dir.exists()
    assert list(out_dir.iterdir()) == []
