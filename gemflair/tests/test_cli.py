from typer.testing import CliRunner

from gemflair import cli

runner = CliRunner()


def test_every_stage_is_a_command():
    out = runner.invoke(cli.app, ["--help"])
    for name in ("prep", "cohorts", "tokenize", "verify-rope", "winnow",
                 "extract", "fit", "report", "run"):
        assert name in out.stdout


def test_run_lists_stages_in_order():
    assert cli.STAGES == ["prep", "cohorts", "tokenize", "winnow", "extract",
                          "fit", "report"]
