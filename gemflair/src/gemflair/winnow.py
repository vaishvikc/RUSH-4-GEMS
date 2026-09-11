import polars as pl

INFERENCE_COLS = ["tokens_past", "s_elapsed_past"]
INDEX_COLS = ["row_ix", "subject_id", "feature_cutoff_dttm", "n_past", "n_kept"]


def cut_points(cohorts):
    return (pl.concat([c.select(subject_id="hospitalization_id",
                                feature_cutoff_dttm="feature_cutoff_dttm")
                       for c in cohorts])
            .unique()
            .sort("subject_id", "feature_cutoff_dttm"))


def count_past(tokens_times, cuts):
    # join_asof finds the last event at or before the cutoff; its index + 1 is the count.
    events = (tokens_times.select("subject_id", "times").explode("times")
              .with_columns(ix=pl.int_range(pl.len()).over("subject_id")))
    return (cuts.sort("feature_cutoff_dttm")
            .join_asof(events.sort("times"), left_on="feature_cutoff_dttm",
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


def write(frame, processed_dir):
    # "held_out" is cotorra's filename slot, not a statistical split.
    frame.select(INFERENCE_COLS).write_parquet(
        processed_dir / "held_out_for_inference.parquet")
    frame.select(INDEX_COLS).write_parquet(processed_dir / "cut_index.parquet")
