import json

import polars as pl
import yaml


def _r(name, ok, detail=""):
    return {"check": name, "ok": bool(ok), "detail": str(detail)}


def startup(cfg):
    from gemflair import config
    gen_dir = config.paths(cfg)["generated"]
    model = json.loads((cfg["model"]["dir"] / "config.json").read_text())
    tkzr = yaml.safe_load(cfg["model"]["tokenizer"].read_text())
    gen = json.loads((cfg["model"]["dir"] / "generation_config.json").read_text())
    clif = json.loads((gen_dir / "clif_config.json").read_text())
    collation = yaml.safe_load((gen_dir / "collation.yaml").read_text())
    return [
        _r("vocab_size_matches_tokenizer",
           model["vocab_size"] == len(tkzr["lookup"]),
           f"{model['vocab_size']} vs {len(tkzr['lookup'])}"),
        _r("bos_eos_match_tokenizer",
           gen["bos_token_id"] == tkzr["lookup"]["BOS"]
           and gen["eos_token_id"] == tkzr["lookup"]["EOS"],
           f"{gen['bos_token_id']}/{gen['eos_token_id']}"),
        _r("max_len_fits_model",
           cfg["model"]["max_len"] <= model["max_position_embeddings"],
           cfg["model"]["max_len"]),
        _r("timezones_agree",
           clif["timezone"] == collation["default_timezone"],
           f"{clif['timezone']} vs {collation['default_timezone']}"),
    ]


def after_winnow(frame, tokens_times, cohorts, max_len):
    # Recompute the last kept event time independently of the winnow query.
    check = (frame.join(tokens_times.select("subject_id", "times"), on="subject_id")
             .with_columns(last=pl.col("times").list.head("n_past").list.last()))
    leaks = check.filter(pl.col("last") > pl.col("feature_cutoff_dttm")).height

    lengths = frame.select(
        bad=(pl.col("tokens_past").list.len() != pl.col("s_elapsed_past").list.len())
        | (pl.col("tokens_past").list.len() > max_len))["bad"].sum()

    kept = frame.select("subject_id", "feature_cutoff_dttm").unique()
    removed = []
    for cohort in cohorts.values():
        missing = cohort.join(kept, left_on=["hospitalization_id", "feature_cutoff_dttm"],
                              right_on=["subject_id", "feature_cutoff_dttm"], how="anti")
        removed += missing["prediction_id"].to_list()

    dupes = frame.height - kept.height
    truncated = frame.filter(pl.col("n_past") > max_len).height
    stitched = sum(c.group_by("hospitalization_join_id")
                   .agg(n=pl.col("hospitalization_id").n_unique())
                   .filter(pl.col("n") > 1).height for c in cohorts.values())
    tokens = tokens_times["tokens"].explode()
    unk = (tokens == 0).sum() / max(1, len(tokens))
    return [
        _r("no_future_data", leaks == 0, f"{leaks} rows past cutoff"),
        _r("token_elapsed_lengths_match", lengths == 0, f"{lengths} bad rows"),
        _r("one_row_per_cut_point", dupes == 0, f"{dupes} duplicate cut points"),
        _r("rows_removed", True, f"{len(removed)} removed: {removed[:20]}"),
        _r("context_truncated", True, f"{truncated} rows exceeded max_len"),
        _r("stitched_encounters", True, f"{stitched} multi-hospitalization encounters"),
        _r("unk_rate", True, f"{unk:.4f} of tokens are UNK"),
    ]


def after_extract(features, index, n_files):
    return [
        _r("single_feature_file", n_files == 1, f"{n_files} files"),
        _r("feature_rows_match_index", features.height == index.height,
           f"{features.height} vs {index.height}"),
    ]


def save(results, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return all(r["ok"] for r in results)
