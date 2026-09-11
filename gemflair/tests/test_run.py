import datetime as dt

import polars as pl

from gemflair import run


def test_raw_view_symlinks_both_directories(tmp_path):
    raw, derived, work = (tmp_path / n for n in ("raw", "derived", "work"))
    for d in (raw, derived):
        d.mkdir()
    (raw / "clif_labs.parquet").write_text("x")
    (derived / "clif_sofa.parquet").write_text("y")
    cfg = {"data": {"clif_dir": raw, "derived_dir": derived}, "work_dir": work}
    view = run.raw_view(cfg)
    assert {p.name for p in view.iterdir()} == {"clif_labs.parquet", "clif_sofa.parquet"}
    assert (view / "clif_labs.parquet").is_symlink()
    assert (view / "clif_sofa.parquet").is_symlink()


def test_raw_view_lets_derived_shadow_raw(tmp_path):
    raw, derived, work = (tmp_path / n for n in ("raw", "derived", "work"))
    for d in (raw, derived):
        d.mkdir()
    (raw / "clif_sofa.parquet").write_text("raw")
    (derived / "clif_sofa.parquet").write_text("derived")
    cfg = {"data": {"clif_dir": raw, "derived_dir": derived}, "work_dir": work}
    view = run.raw_view(cfg)
    assert (view / "clif_sofa.parquet").resolve() == (derived / "clif_sofa.parquet")


def _cohort():
    return pl.DataFrame({
        "hospitalization_join_id": ["e1", "e1", "e2", "e3", "e4"],
        "split": ["train", "train", "train", "test", "test"],
        "feature_cutoff_dttm": [dt.datetime(2020, 1, 1)] * 5,
    })


def test_subsample_none_returns_everything():
    c = _cohort()
    assert run.subsample(c, None, seed=42).equals(c)


def _cohort_with_two_multirow_encounters():
    return pl.DataFrame({
        "hospitalization_join_id": ["e1", "e1", "e2", "e2", "e3", "e4"],
        "split": ["train", "train", "train", "train", "test", "test"],
        "feature_cutoff_dttm": [dt.datetime(2020, 1, 1)] * 6,
    })


def test_subsample_keeps_whole_encounters():
    cohort = _cohort_with_two_multirow_encounters()
    out = run.subsample(cohort, n=1, seed=42)
    for enc in out["hospitalization_join_id"].unique():
        original = cohort.filter(pl.col("hospitalization_join_id") == enc).height
        assert out.filter(pl.col("hospitalization_join_id") == enc).height == original
    assert out.filter(pl.col("hospitalization_join_id") == "e2").height == 2


def test_subsample_keeps_both_splits():
    out = run.subsample(_cohort(), n=1, seed=42)
    assert set(out["split"].unique()) == {"train", "test"}


def test_subsample_is_reproducible():
    a = run.subsample(_cohort(), n=1, seed=7)
    b = run.subsample(_cohort(), n=1, seed=7)
    assert a.equals(b)
