import datetime as dt
import json

import polars as pl

from gemflair import audit, winnow

H = lambda *hours: [dt.datetime(2020, 1, 1, h) for h in hours]
AT = lambda h: dt.datetime(2020, 1, 1, h)

TOKENS = pl.DataFrame({
    "subject_id": ["a"],
    "tokens": [[10, 11, 12, 13]],
    "times": [H(1, 2, 3, 4)],
})
COHORT = pl.DataFrame({
    "hospitalization_id": ["a", "a"],
    "hospitalization_join_id": ["e1", "e1"],
    "prediction_id": ["p1", "p2"],
    "feature_cutoff_dttm": [AT(2), AT(4)],
})


def _ok(results, name):
    return next(r["ok"] for r in results if r["check"] == name)


def test_no_leakage_passes_on_a_correct_cut():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    res = audit.after_winnow(frame, TOKENS, {"t": COHORT}, max_len=10)
    assert _ok(res, "no_future_data")


def test_no_leakage_fails_when_a_cutoff_is_moved_backwards():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    tampered = frame.with_columns(
        feature_cutoff_dttm=pl.lit(AT(0)).cast(frame["feature_cutoff_dttm"].dtype))
    res = audit.after_winnow(tampered, TOKENS, {"t": COHORT}, max_len=10)
    assert not _ok(res, "no_future_data")


def test_length_check_fails_on_mismatched_lists():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    tampered = frame.with_columns(s_elapsed_past=pl.lit([1.0]))
    res = audit.after_winnow(tampered, TOKENS, {"t": COHORT}, max_len=10)
    assert not _ok(res, "token_elapsed_lengths_match")


def test_removed_rows_are_reported_with_prediction_ids():
    cohort = COHORT.with_columns(feature_cutoff_dttm=pl.Series([AT(0), AT(2)]))
    frame = winnow.cut(TOKENS, winnow.cut_points([cohort]), max_len=10)
    res = audit.after_winnow(frame, TOKENS, {"t": cohort}, max_len=10)
    removed = next(r for r in res if r["check"] == "rows_removed")
    assert "p1" in removed["detail"]


def test_duplicate_cut_points_are_caught():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    doubled = pl.concat([frame, frame])
    res = audit.after_winnow(doubled, TOKENS, {"t": COHORT}, max_len=10)
    assert not _ok(res, "one_row_per_cut_point")


def test_unk_rate_is_reported():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    res = audit.after_winnow(frame, TOKENS, {"t": COHORT}, max_len=10)
    assert next(r for r in res if r["check"] == "unk_rate")["detail"].endswith("UNK")


def test_after_extract_fails_on_multiple_feature_files():
    idx = pl.DataFrame({"row_ix": [0, 1]})
    feats = pl.DataFrame({"features": [[0.0], [0.0]]})
    res = audit.after_extract(feats, idx, n_files=2)
    assert not _ok(res, "single_feature_file")


def test_after_extract_fails_on_row_count_mismatch():
    idx = pl.DataFrame({"row_ix": [0, 1]})
    feats = pl.DataFrame({"features": [[0.0]]})
    res = audit.after_extract(feats, idx, n_files=1)
    assert not _ok(res, "feature_rows_match_index")


def test_save_returns_false_and_writes_json(tmp_path):
    out = tmp_path / "a.json"
    assert audit.save([{"check": "x", "ok": False, "detail": "d"}], out) is False
    assert json.loads(out.read_text())[0]["check"] == "x"
