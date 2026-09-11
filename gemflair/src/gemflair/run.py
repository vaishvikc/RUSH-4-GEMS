import subprocess

import numpy as np
import polars as pl

from gemflair import config


def _sh(*args):
    subprocess.run([str(a) for a in args], check=True)


def raw_view(cfg):
    # cocoa takes one --raw-data-home, so link both directories into one view.
    view = config.paths(cfg)["raw_view"]
    view.mkdir(parents=True, exist_ok=True)
    for src in (cfg["data"]["clif_dir"], cfg["data"]["derived_dir"]):
        for f in sorted(src.glob("*.parquet")) if src.is_dir() else []:
            link = view / f.name
            link.unlink(missing_ok=True)
            link.symlink_to(f)
    return view


def subsample(cohort, n, seed):
    if n is None:
        return cohort
    rng = np.random.default_rng(seed)
    keep = []
    for split in sorted(cohort["split"].unique()):
        enc = cohort.filter(pl.col("split") == split)["hospitalization_join_id"].unique().sort()
        keep += list(rng.choice(enc.to_numpy(), min(n, len(enc)), replace=False))
    return cohort.filter(pl.col("hospitalization_join_id").is_in(keep))


def load_task(cfg, task):
    p = config.paths(cfg)
    out = p["cohorts"] / f"{task}.parquet"
    args = ["flair", "load-task", task,
            "--clif-config", p["generated"] / "clif_config.json", "--out", out]
    if cfg["split"]["train_end"] and cfg["split"]["test_start"]:
        args += ["--train-end", cfg["split"]["train_end"],
                 "--test-start", cfg["split"]["test_start"]]
    _sh(*args)
    trimmed = subsample(pl.read_parquet(out),
                        cfg["limits"]["max_encounters_per_task"], cfg["xgboost"]["seed"])
    trimmed.write_parquet(out)
    return out


def collate(cfg):
    p = config.paths(cfg)
    _sh("cocoa", "collate", "--collation-config", p["generated"] / "collation.yaml",
        "--raw-data-home", raw_view(cfg), "--processed-data-home", p["processed"])


def tokenize(cfg):
    p = config.paths(cfg)
    _sh("cocoa", "tokenize", "--tokenizer-home", cfg["model"]["tokenizer"],
        "--processed-data-home", p["processed"])


def extract(cfg):
    p = config.paths(cfg)
    _sh("cotorra", "extract", "--extraction-config", p["generated"] / "extraction.yaml",
        "--processed-data-home", p["processed"], "--model-home", cfg["model"]["dir"],
        "--output-home", p["processed"])
    return sorted(p["processed"].glob("features-held_out-*.parquet"))


def build_report(cfg, task):
    p = config.paths(cfg)
    out = p["reports"] / task
    _sh("flair", "build-report", p["preds"] / f"{task}.parquet", "--task", task,
        "--cohort", p["cohorts"] / f"{task}.parquet", "--out", out)
    return out
