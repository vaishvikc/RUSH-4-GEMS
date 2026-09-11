import datetime as dt

import polars as pl
import pytest

from gemflair import winnow

H = lambda *hours: [dt.datetime(2020, 1, 1, h) for h in hours]
AT = lambda h: dt.datetime(2020, 1, 1, h)


@pytest.fixture
def tokens_times():
    return pl.DataFrame({
        "subject_id": ["a", "b"],
        "tokens": [[10, 11, 12, 13], [20, 21]],
        "times": [H(1, 2, 3, 4), H(1, 5)],
    })


def _cuts(pairs):
    return pl.DataFrame(
        {"subject_id": [s for s, _ in pairs],
         "feature_cutoff_dttm": [AT(h) for _, h in pairs]})


def test_cutoff_is_inclusive(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 2)]), max_len=10)
    assert out["tokens_past"].to_list() == [[10, 11]]


def test_keeps_last_max_len_not_first(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 9)]), max_len=3)
    assert out["tokens_past"].to_list() == [[11, 12, 13]]
    assert out["n_past"].to_list() == [4]
    assert out["n_kept"].to_list() == [3]


def test_s_elapsed_is_not_rebased_after_trimming(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 9)]), max_len=3)
    assert out["s_elapsed_past"].to_list()[0][0] == 3600.0


def test_cut_happens_before_trim(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 3)]), max_len=2)
    assert out["tokens_past"].to_list() == [[11, 12]]
    assert out["s_elapsed_past"].to_list() == [[3600.0, 7200.0]]


def test_cutoff_before_first_event_is_dropped(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("b", 0)]), max_len=10)
    assert out.height == 0


def test_one_subject_many_cutpoints_are_nested(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 2), ("a", 3), ("a", 4)]), max_len=10)
    assert out.height == 3
    got = sorted(out["tokens_past"].to_list(), key=len)
    assert got == [[10, 11], [10, 11, 12], [10, 11, 12, 13]]


def test_row_ix_is_contiguous_from_zero(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 2), ("b", 5), ("a", 4)]), max_len=10)
    assert out["row_ix"].to_list() == list(range(out.height))


def test_token_and_elapsed_lengths_always_match(tokens_times):
    out = winnow.cut(tokens_times, _cuts([("a", 2), ("a", 9), ("b", 5)]), max_len=3)
    lens = zip(out["tokens_past"].to_list(), out["s_elapsed_past"].to_list())
    assert all(len(t) == len(s) for t, s in lens)


def test_cut_points_deduplicate_across_tasks():
    one = pl.DataFrame({"hospitalization_id": ["a", "a"],
                        "feature_cutoff_dttm": [AT(2), AT(3)]})
    two = pl.DataFrame({"hospitalization_id": ["a", "b"],
                        "feature_cutoff_dttm": [AT(2), AT(5)]})
    out = winnow.cut_points([one, two])
    assert out.height == 3


def test_write_keeps_both_files_in_the_same_order(tokens_times, tmp_path):
    out = winnow.cut(tokens_times, _cuts([("a", 2), ("b", 5), ("a", 4)]), max_len=10)
    winnow.write(out, tmp_path)
    inf = pl.read_parquet(tmp_path / "held_out_for_inference.parquet")
    idx = pl.read_parquet(tmp_path / "cut_index.parquet")
    assert inf.height == idx.height == out.height
    assert idx["row_ix"].to_list() == list(range(out.height))
    assert inf["tokens_past"].to_list() == out["tokens_past"].to_list()
