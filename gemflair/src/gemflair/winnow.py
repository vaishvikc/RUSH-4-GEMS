import hashlib

import polars as pl

INFERENCE_COLS = ["tokens_past", "s_elapsed_past"]
INDEX_COLS = ["row_ix", "subject_id", "feature_cutoff_dttm", "n_past", "n_kept"]


def cut_points(cohorts):
    return (pl.concat([c.select(
                            subject_id="hospitalization_id",
                            feature_cutoff_dttm=pl.col("feature_cutoff_dttm")
                            .dt.cast_time_unit("us"))
                       for c in cohorts])
            .unique()
            .sort("subject_id", "feature_cutoff_dttm"))


def count_past(tokens_times, cuts):
    # join_asof finds the last event at or before the cutoff; its index + 1 is the count.
    events = (tokens_times.select("subject_id", "times").explode("times")
              .with_columns(ix=pl.int_range(pl.len()).over("subject_id")))
    time_dtype = events.schema["times"]
    cutoff_dtype = cuts.schema["feature_cutoff_dttm"]
    cutoff = pl.col("feature_cutoff_dttm")
    if time_dtype.time_zone and not cutoff_dtype.time_zone:
        localized = cutoff.dt.replace_time_zone(
            time_dtype.time_zone, ambiguous="latest", non_existent="null")
        shifted = (cutoff + pl.duration(hours=1)).dt.replace_time_zone(
            time_dtype.time_zone, ambiguous="latest", non_existent="null")
        cutoff = pl.coalesce(localized, shifted)
    elif time_dtype.time_zone:
        cutoff = cutoff.dt.convert_time_zone(time_dtype.time_zone)
    cutoff = cutoff.dt.cast_time_unit(time_dtype.time_unit)
    keyed = cuts.with_columns(cutoff.alias("_cutoff_key"))
    return (keyed.sort("_cutoff_key")
            .join_asof(events.sort("times"), left_on="_cutoff_key",
                       right_on="times", by="subject_id", strategy="backward")
            .with_columns(n_past=(pl.col("ix") + 1).fill_null(0))
            .select("subject_id", "feature_cutoff_dttm", "n_past"))


def cut(tokens_times, cuts, max_len):
    n = count_past(tokens_times, cuts)
    return (tokens_times.join(n, on="subject_id")
            .filter(pl.col("n_past") > 0)
            .with_columns(
                tokens_past=pl.col("tokens").list.head("n_past").list.tail(max_len),
                s_elapsed_past=pl.col("times").list.eval(
                    (pl.element() - pl.element().first()).dt.total_seconds()
                ).list.head("n_past").list.tail(max_len))
            .with_columns(n_kept=pl.col("tokens_past").list.len())
            .sort("subject_id", "feature_cutoff_dttm")
            .with_row_index("row_ix")
            .select(INDEX_COLS + INFERENCE_COLS))


def fingerprint(frame):
    ident = frame.select("subject_id", "feature_cutoff_dttm")
    rows = "\n".join(f"{s}|{t}" for s, t in ident.iter_rows())
    return hashlib.sha256(rows.encode()).hexdigest()


def write(frame, processed_dir):
    # "held_out" is cotorra's filename slot, not a statistical split.
    frame.select(INFERENCE_COLS).write_parquet(
        processed_dir / "held_out_for_inference.parquet")
    frame.select(INDEX_COLS).write_parquet(processed_dir / "cut_index.parquet")
    (processed_dir / "cut_index.fingerprint").write_text(fingerprint(frame))
