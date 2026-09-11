import datetime as dt

import numpy as np
import polars as pl

from gemflair import fit

AT = lambda h: dt.datetime(2020, 1, 1, h)

INDEX = pl.DataFrame({
    "row_ix": [7, 2, 9],
    "subject_id": ["a", "a", "b"],
    "feature_cutoff_dttm": [AT(2), AT(4), AT(2)],
})
FEATURES = pl.DataFrame({"features": [[float(i), float(i) + 100.0] for i in range(10)]})
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
    # row_ix values are non-contiguous [7, 2, 9]; sorted order is row_ix 2, 7, 9.
    assert X.tolist() == [[2.0, 102.0], [7.0, 107.0], [9.0, 109.0]]


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


def test_val_mask_is_subset_of_train_and_selects_something():
    df = pl.DataFrame({"hospitalization_join_id": ["e1", "e1", "e2", "e3", "e_out"]})
    train = np.array([True, True, True, True, False])
    mask = fit.val_mask(df, train, seed=1)
    assert not (mask & ~train).any()
    assert mask.any()


def _synthetic(label_col):
    subj, enc, pred, cutoff, split, label, feats = [], [], [], [], [], [], []
    rng = np.random.default_rng(0)
    row = 0
    plan = [("e1", "train", 4), ("e2", "train", 4), ("e3", "train", 4),
            ("e4", "test", 3), ("e5", "test", 3)]
    for enc_name, sp, n in plan:
        for i in range(n):
            subj.append(f"s{row}")
            enc.append(enc_name)
            pred.append(f"pr{row}")
            cutoff.append(AT(row % 20 + 1))
            split.append(sp)
            label.append(i % 2)
            feats.append(rng.normal(size=3).tolist())
            row += 1
    index = pl.DataFrame({"row_ix": list(range(row)), "subject_id": subj,
                          "feature_cutoff_dttm": cutoff})
    features = pl.DataFrame({"features": feats})
    cohort = pl.DataFrame({"hospitalization_id": subj, "hospitalization_join_id": enc,
                           "prediction_id": pred, "feature_cutoff_dttm": cutoff,
                           "split": split, label_col: label})
    return cohort, index, features


def test_run_produces_aligned_valid_probabilities_with_disjoint_val_split():
    cohort, index, features = _synthetic("label")
    cfg = {"xgboost": {"seed": 0, "early_stopping_rounds": 2}}
    df_before, _ = fit.attach(cohort, index, features)
    out = fit.run(cohort, index, features, "label", cfg)

    assert out.height == df_before.height
    before_ids = df_before["prediction_id"].to_list()
    after_ids = out["prediction_id"].to_list()
    assert sorted(after_ids) == sorted(before_ids)
    assert len(after_ids) == len(set(after_ids))

    assert "y_prob" in out.columns
    y_prob = out["y_prob"].to_numpy()
    assert ((y_prob >= 0) & (y_prob <= 1)).all()

    train = (df_before["split"] == "train").to_numpy()
    val = fit.val_mask(df_before, train, cfg["xgboost"]["seed"]) & train
    fitted = train & ~val
    assert (val <= train).all()
    assert not (val & fitted).any()
