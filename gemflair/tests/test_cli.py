from typer.testing import CliRunner

from gemflair import cli

runner = CliRunner()


def test_every_stage_is_a_command():
    out = runner.invoke(cli.app, ["--help"])
    for name in ("prep", "cohorts", "tokenize", "verify-rope", "winnow",
                 "extract", "fit", "report", "plot-auroc", "run"):
        assert name in out.stdout


def test_run_lists_stages_in_order():
    assert cli.STAGES == ["prep", "cohorts", "tokenize", "winnow", "extract",
                          "fit", "report"]


def _p(processed_dir):
    return {"processed": processed_dir}


def test_features_stale_when_fingerprints_match(tmp_path):
    (tmp_path / "cut_index.fingerprint").write_text("abc")
    (tmp_path / "features.fingerprint").write_text("abc")
    assert not cli._features_stale(_p(tmp_path))


def test_features_stale_when_fingerprints_differ(tmp_path):
    (tmp_path / "cut_index.fingerprint").write_text("abc")
    (tmp_path / "features.fingerprint").write_text("xyz")
    assert cli._features_stale(_p(tmp_path))


def test_features_stale_when_features_fingerprint_missing(tmp_path):
    (tmp_path / "cut_index.fingerprint").write_text("abc")
    assert cli._features_stale(_p(tmp_path))


def test_features_stale_when_cut_index_fingerprint_missing(tmp_path):
    (tmp_path / "features.fingerprint").write_text("abc")
    assert cli._features_stale(_p(tmp_path))


def test_features_file_returns_none_when_extract_never_ran(tmp_path):
    assert cli._features_file(_p(tmp_path)) is None


def test_features_file_finds_the_extracted_parquet(tmp_path):
    f = tmp_path / "features-held_out-mygem.parquet"
    f.write_text("x")
    assert cli._features_file(_p(tmp_path)) == f
