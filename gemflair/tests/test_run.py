import datetime as dt

import polars as pl
import pytest

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


def test_collator_shifts_nonexistent_dst_time_forward():
    collator = object.__new__(run._DstSafeCollator)
    collator.tz = "US/Central"
    frame = pl.LazyFrame({"time": [dt.datetime(2018, 3, 11, 1),
                                    dt.datetime(2018, 3, 11, 2)]})
    result = frame.select(collator.to_default_tz(frame, "time")).collect()["time"]
    assert [value.hour for value in result] == [1, 3]
    assert str(result.dtype.time_zone) == "US/Central"


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


def test_load_task_skips_the_subprocess_when_the_cohort_already_exists(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "cohorts").mkdir(parents=True)
    existing = work / "cohorts" / "icu_daily_mortality.parquet"
    cohort = _cohort_with_two_multirow_encounters()
    cohort.write_parquet(existing)
    monkeypatch.setattr(run, "_sh", lambda *a: pytest.fail("flair load-task should not run"))
    cfg = {"work_dir": work, "limits": {"max_encounters_per_task": None},
           "xgboost": {"seed": 42}}
    out = run.load_task(cfg, "icu_daily_mortality")
    assert out == existing
    assert pl.read_parquet(existing).equals(cohort)


def test_load_task_reapplies_the_cap_to_a_cached_cohort_without_the_subprocess(
        tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "cohorts").mkdir(parents=True)
    existing = work / "cohorts" / "icu_daily_mortality.parquet"
    _cohort().write_parquet(existing)
    monkeypatch.setattr(run, "_sh", lambda *a: pytest.fail("flair load-task should not run"))
    cfg = {"work_dir": work, "limits": {"max_encounters_per_task": 1},
           "xgboost": {"seed": 42}}
    out = run.load_task(cfg, "icu_daily_mortality")
    result = pl.read_parquet(out)
    assert result["hospitalization_join_id"].n_unique() == 2
    assert set(result["split"].unique()) == {"train", "test"}


def test_shard_bounds_cover_odd_size_contiguously():
    assert run.shard_bounds(5, 2, 0) == (0, 3)
    assert run.shard_bounds(5, 2, 1) == (3, 5)


def test_merge_feature_parts_preserves_row_order(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    final = tmp_path / "features.parquet"
    pl.DataFrame({"_row_ix": [0, 1, 2],
                  "features": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]}).write_parquet(first)
    pl.DataFrame({"_row_ix": [3, 4],
                  "features": [[3.0, 3.0], [4.0, 4.0]]}).write_parquet(second)
    run.merge_feature_parts([second, first], final, total=5, feature_size=2)
    assert pl.read_parquet(final)["features"].to_list() == [
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]


def test_merge_feature_parts_rejects_missing_rows(tmp_path):
    part = tmp_path / "part.parquet"
    pl.DataFrame({"_row_ix": [0, 2],
                  "features": [[0.0, 0.0], [2.0, 2.0]]}).write_parquet(part)
    with pytest.raises(RuntimeError, match="exactly once"):
        run.merge_feature_parts([part], tmp_path / "features.parquet",
                                total=3, feature_size=2)
