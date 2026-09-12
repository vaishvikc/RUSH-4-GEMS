import json
import pathlib

import yaml

from gemflair import config

HERE = pathlib.Path(__file__).parent


def _cfg(tmp_path):
    raw = yaml.safe_load((HERE.parent / "config" / "gemflair.yaml").read_text())
    raw["work_dir"] = str(tmp_path / "work")
    p = tmp_path / "gemflair.yaml"
    p.write_text(yaml.safe_dump(raw))
    return config.load(p)


def test_load_resolves_paths(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg["work_dir"].is_absolute()
    assert cfg["model"]["dir"].is_absolute()
    assert cfg["model"]["trainer_state"] is None


def test_paths_are_under_work_dir(tmp_path):
    cfg = _cfg(tmp_path)
    for p in config.paths(cfg).values():
        assert cfg["work_dir"] in p.parents or p == cfg["work_dir"]


def test_clif_config_matches_flair_schema(tmp_path):
    cfg = _cfg(tmp_path)
    out = config.write_clif_config(cfg, tmp_path / "clif.json")
    doc = json.loads(out.read_text())
    assert doc["site"] == "rush"
    assert doc["filetype"] == "parquet"
    assert doc["timezone"] == "US/Central"


def test_collation_timezone_follows_config(tmp_path):
    cfg = _cfg(tmp_path)
    out = config.write_collation_config(cfg, tmp_path / "collation.yaml")
    doc = yaml.safe_load(out.read_text())
    assert doc["default_timezone"] == "US/Central"
    assert doc["subject_id"] == "hospitalization_id"


def test_extraction_config_never_sets_shard_size(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg["extract"]["devices"] == [0, 1]
    cfg["model"]["time_based_rope"] = {"sec_per_pos_id": 300}
    out = config.write_extraction_config(cfg, tmp_path / "extraction.yaml")
    doc = yaml.safe_load(out.read_text())
    assert "shard_size" not in doc["extract"]
    assert doc["time_based_rope"]["sec_per_pos_id"] == 300


def test_extraction_config_omits_rope_when_null(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["model"]["time_based_rope"] = None
    out = config.write_extraction_config(cfg, tmp_path / "extraction.yaml")
    assert "time_based_rope" not in yaml.safe_load(out.read_text())
