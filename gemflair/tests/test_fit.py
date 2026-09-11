import datetime as dt

import numpy as np
import polars as pl

from gemflair import fit

AT = lambda h: dt.datetime(2020, 1, 1, h)

INDEX = pl.DataFrame({
    "row_ix": [0, 1, 2],
    "subject_id": ["a", "a", "b"],
    "feature_cutoff_dttm": [AT(2), AT(4), AT(2)],
})
FEATURES = pl.DataFrame({"features": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]})
COHORT = pl.DataFrame({
    "hospitalization_id": ["a", "a", "b", "b"],
    "hospitalization_join_id": ["e1", "e1", "e2", "e2"],
    "prediction_id": ["p1", "p2", "p3", "p4"],
    "feature_cutoff_dttm": [AT(2), AT(4), AT(2), AT(9)],
    "split": ["train", "train", "test", "test"],
    "label_x": [0, 1, 0, 1],
})


def test_attach_selects_features_by_row_ix():
    df, X = fit.attach(COHORT, INDEX, FEATURES)
    assert df.height == 3
    assert X.shape == (3, 2)
    assert X[0].tolist() == [1.0, 0.0]


def test_attach_drops_cohort_rows_with_no_representation():
    df, _ = fit.attach(COHORT, INDEX, FEATURES)
    assert "p4" not in df["prediction_id"].to_list()


def test_attach_returns_float32():
    _, X = fit.attach(COHORT, INDEX, FEATURES)
    assert X.dtype == np.float32


def test_val_mask_holds_out_whole_encounters():
    df = pl.DataFrame({"hospitalization_join_id": ["e1", "e1", "e2", "e3"]})
    train = np.array([True, True, True, True])
    mask = fit.val_mask(df, train, seed=42)
    for enc in ("e1", "e2", "e3"):
        rows = [m for m, e in zip(mask, df["hospitalization_join_id"]) if e == enc]
        assert len(set(rows)) == 1


def test_val_mask_is_reproducible():
    df = pl.DataFrame({"hospitalization_join_id": [f"e{i}" for i in range(20)]})
    train = np.ones(20, dtype=bool)
    assert fit.val_mask(df, train, 7).tolist() == fit.val_mask(df, train, 7).tolist()
