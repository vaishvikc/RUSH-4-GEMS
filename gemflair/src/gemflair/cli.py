import json

import numpy as np
import polars as pl
import typer

from gemflair import audit, config, fit, prep, run, verify_rope, winnow

app = typer.Typer(add_completion=False)
CONFIG = typer.Option("config/gemflair.yaml", "--config", "-c")
STAGES = ["prep", "cohorts", "tokenize", "winnow", "extract", "fit", "report"]


def _setup(path):
    cfg = config.load(path)
    p = config.paths(cfg)
    for d in p.values():
        d.mkdir(parents=True, exist_ok=True)
    config.write_clif_config(cfg, p["generated"] / "clif_config.json")
    config.write_collation_config(cfg, p["generated"] / "collation.yaml")
    config.write_extraction_config(cfg, p["generated"] / "extraction.yaml")
    if not audit.save(audit.startup(cfg), p["audit"] / "startup.json"):
        raise typer.Exit(1)
    return cfg, p


def _cohorts(cfg, p):
    return {t: pl.read_parquet(p["cohorts"] / f"{t}.parquet") for t in cfg["tasks"]}


@app.command("prep")
def prep_cmd(config_path: str = CONFIG):
    cfg, _ = _setup(config_path)
    prep.build(cfg)


@app.command()
def cohorts(config_path: str = CONFIG):
    cfg, _ = _setup(config_path)
    for task in cfg["tasks"]:
        run.load_task(cfg, task)


@app.command()
def tokenize(config_path: str = CONFIG):
    cfg, _ = _setup(config_path)
    run.collate(cfg)
    run.tokenize(cfg)


@app.command("verify-rope")
def verify_rope_cmd(config_path: str = CONFIG):
    cfg, p = _setup(config_path)
    result = verify_rope.run(cfg)
    (p["audit"] / "verify_rope.json").write_text(json.dumps(result, indent=2))
    typer.echo(json.dumps(result, indent=2))


@app.command("winnow")
def winnow_cmd(config_path: str = CONFIG):
    cfg, p = _setup(config_path)
    cohorts = _cohorts(cfg, p)
    tt = pl.read_parquet(p["processed"] / "tokens_times.parquet")
    frame = winnow.cut(tt, winnow.cut_points(list(cohorts.values())),
                       cfg["model"]["max_len"])
    winnow.write(frame, p["processed"])
    _write_subject_splits(cohorts, p, cfg["xgboost"]["seed"])
    if not audit.save(audit.after_winnow(frame, tt, cohorts, cfg["model"]["max_len"]),
                      p["audit"] / "winnow.json"):
        raise typer.Exit(1)


def _write_subject_splits(cohorts, p, seed):
    # cotorra's Loader needs this file; extraction ignores the split values, but
    # verify-rope reads the tuning split, so carve one out of train.
    df = (pl.concat([c.select(subject_id="hospitalization_id", split="split")
                     for c in cohorts.values()])
          .unique(subset=["subject_id"])
          .with_columns(split=pl.when(pl.col("split") == "test")
                        .then(pl.lit("held_out")).otherwise(pl.lit("train"))))
    train = df.filter(pl.col("split") == "train")["subject_id"]
    rng = np.random.default_rng(seed)
    tuning = set(rng.choice(train.sort().to_numpy(),
                            max(1, round(0.1 * len(train))), replace=False))
    df = df.with_columns(split=pl.when(pl.col("subject_id").is_in(tuning))
                         .then(pl.lit("tuning")).otherwise(pl.col("split")))
    df.write_parquet(p["processed"] / "subject_splits.parquet")


@app.command()
def extract(config_path: str = CONFIG):
    cfg, p = _setup(config_path)
    files = run.extract(cfg)
    index = pl.read_parquet(p["processed"] / "cut_index.parquet")
    feats = pl.read_parquet(files[0]) if files else pl.DataFrame({"features": []})
    if not audit.save(audit.after_extract(feats, index, len(files)),
                      p["audit"] / "extract.json"):
        raise typer.Exit(1)


@app.command("fit")
def fit_cmd(config_path: str = CONFIG):
    from flair_benchmark.tasks import get_task
    cfg, p = _setup(config_path)
    index = pl.read_parquet(p["processed"] / "cut_index.parquet")
    feats = pl.read_parquet(next(p["processed"].glob("features-held_out-*.parquet")))
    for task, cohort in _cohorts(cfg, p).items():
        label = get_task(task).META["label_column"]
        fit.run(cohort, index, feats, label, cfg).write_parquet(
            p["preds"] / f"{task}.parquet")


@app.command()
def report(config_path: str = CONFIG):
    cfg, _ = _setup(config_path)
    for task in cfg["tasks"]:
        run.build_report(cfg, task)


@app.command("run")
def run_all(config_path: str = CONFIG):
    for stage in STAGES:
        typer.echo(f"== {stage}")
        {"prep": prep_cmd, "cohorts": cohorts, "tokenize": tokenize,
         "winnow": winnow_cmd, "extract": extract, "fit": fit_cmd,
         "report": report}[stage](config_path)


def main():
    app()
