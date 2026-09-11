import datetime as dt
import json

import polars as pl
import yaml

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


def test_no_future_data_fails_when_n_past_is_undercounted():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    tampered = frame.with_columns(
        n_past=pl.when(pl.col("feature_cutoff_dttm") == AT(4))
        .then(pl.col("n_past") - 1).otherwise(pl.col("n_past")))
    res = audit.after_winnow(tampered, TOKENS, {"t": COHORT}, max_len=10)
    assert not _ok(res, "no_future_data")


def test_length_check_fails_when_tokens_exceed_max_len():
    frame = winnow.cut(TOKENS, winnow.cut_points([COHORT]), max_len=10)
    tampered = frame.with_columns(
        tokens_past=pl.lit(list(range(15))),
        s_elapsed_past=pl.lit([float(i) for i in range(15)]))
    res = audit.after_winnow(tampered, TOKENS, {"t": COHORT}, max_len=10)
    assert not _ok(res, "token_elapsed_lengths_match")


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


def test_save_returns_true_when_all_pass(tmp_path):
    out = tmp_path / "b.json"
    assert audit.save([{"check": "x", "ok": True, "detail": "d"}], out) is True


def _startup_cfg(tmp_path, *, vocab_size=5, max_pos=100, max_len=10, bos=0, eos=1,
                  tkzr_bos=0, tkzr_eos=1, clif_tz="US/Eastern", collation_tz="US/Eastern",
                  lookup_size=5):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    gen_dir = tmp_path / "work" / "generated"
    gen_dir.mkdir(parents=True)
    lookup = {"BOS": tkzr_bos, "EOS": tkzr_eos}
    i = 0
    while len(lookup) < lookup_size:
        lookup.setdefault(f"t{i}", i + 2)
        i += 1
    (model_dir / "config.json").write_text(
        json.dumps({"vocab_size": vocab_size, "max_position_embeddings": max_pos}))
    (model_dir / "generation_config.json").write_text(
        json.dumps({"bos_token_id": bos, "eos_token_id": eos}))
    (model_dir / "tokenizer.yaml").write_text(yaml.safe_dump({"lookup": lookup}))
    (gen_dir / "clif_config.json").write_text(json.dumps({"timezone": clif_tz}))
    (gen_dir / "collation.yaml").write_text(yaml.safe_dump({"default_timezone": collation_tz}))
    return {"work_dir": tmp_path / "work",
            "model": {"dir": model_dir, "tokenizer": model_dir / "tokenizer.yaml", "max_len": max_len}}


def test_startup_passes_when_everything_agrees(tmp_path):
    cfg = _startup_cfg(tmp_path)
    res = audit.startup(cfg)
    assert all(r["ok"] for r in res)


def test_startup_fails_on_vocab_size_mismatch(tmp_path):
    cfg = _startup_cfg(tmp_path, vocab_size=999)
    res = audit.startup(cfg)
    assert not _ok(res, "vocab_size_matches_tokenizer")


def test_startup_fails_on_bos_eos_mismatch(tmp_path):
    cfg = _startup_cfg(tmp_path, tkzr_bos=42)
    res = audit.startup(cfg)
    assert not _ok(res, "bos_eos_match_tokenizer")


def test_startup_fails_when_max_len_exceeds_model(tmp_path):
    cfg = _startup_cfg(tmp_path, max_len=1000)
    res = audit.startup(cfg)
    assert not _ok(res, "max_len_fits_model")


def test_startup_fails_on_timezone_mismatch(tmp_path):
    cfg = _startup_cfg(tmp_path, collation_tz="UTC")
    res = audit.startup(cfg)
    assert not _ok(res, "timezones_agree")
