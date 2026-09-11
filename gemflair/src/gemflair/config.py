import json
import pathlib

import yaml

PATH_KEYS = [("work_dir",), ("data", "clif_dir"), ("data", "derived_dir"),
             ("model", "dir"), ("model", "tokenizer"), ("model", "trainer_state")]


def load(path="config/gemflair.yaml"):
    cfg = yaml.safe_load(pathlib.Path(path).read_text())
    for keys in PATH_KEYS:
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = pathlib.Path(d[keys[-1]]).expanduser().resolve()
    return cfg


def paths(cfg):
    w = cfg["work_dir"]
    names = ["raw_view", "processed", "cohorts", "preds", "reports", "audit",
             "generated", "flair_cache"]
    return {n: w / n for n in names}


def write_clif_config(cfg, out):
    doc = {
        "site": cfg["data"]["site"],
        "data_directory": str(cfg["data"]["clif_dir"]),
        "filetype": "parquet",
        "timezone": cfg["data"]["timezone"],
        "stitch_time_interval_hours": 6,
        "cache_directory": str(paths(cfg)["flair_cache"]),
    }
    out.write_text(json.dumps(doc, indent=2))
    return out


def write_collation_config(cfg, out):
    import cocoa.config
    src = pathlib.Path(cocoa.config.__file__).parent / "collation.yaml"
    doc = yaml.safe_load(src.read_text())
    doc["default_timezone"] = cfg["data"]["timezone"]
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out


def write_extraction_config(cfg, out):
    # shard_size is deliberately absent: sharding interleaves rows and breaks
    # the positional join back to cut_index.
    doc = {
        "max_seq_len": cfg["model"]["max_len"],
        "extract": {"max_len": cfg["model"]["max_len"],
                    "batch_size": cfg["extract"]["batch_size"]},
    }
    if cfg["model"]["time_based_rope"]:
        doc["time_based_rope"] = cfg["model"]["time_based_rope"]
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out
