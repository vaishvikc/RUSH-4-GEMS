import json

import matplotlib.pyplot as plt
import pytest

from gemflair import plotting


def _write_report(root, task, mode, body):
    path = root / task / ("landmark.json" if mode == "continuous"
                          else "overall.json")
    path.parent.mkdir(parents=True)
    body["metadata"] = {"task": task, "report_mode": mode, "split": "test"}
    path.write_text(json.dumps(body))
    return path


def _reports(tmp_path):
    for task, auc in (("icu_daily_mortality", 0.81),
                      ("icu_daily_ltach", 0.72)):
        _write_report(tmp_path, task, "continuous", {
            "pooled": {"auroc": auc, "auroc_ci": [auc - 0.02, auc + 0.02]},
            "by_landmark": [
                {"hospitalization_time": 1, "auroc": auc - 0.01,
                 "auroc_ci": [auc - 0.03, auc + 0.01]},
                {"hospitalization_time": 2, "auroc": auc + 0.01,
                 "auroc_ci": [auc - 0.01, auc + 0.03]},
            ],
        })
    for task, auc in (("extubation_failure_24h", 0.63),
                      ("icu_readmission", 0.6365)):
        _write_report(tmp_path, task, "episodic", {
            "discrimination": {
                "auroc": auc,
                "auroc_ci": [auc - 0.03, auc + 0.03],
                "tpr_at_grid": [0, 0.3, 0.7, 1],
            },
        })
    return tmp_path


def test_builds_four_canonical_panels_in_order(tmp_path):
    fig = plotting.build_auroc_figure(_reports(tmp_path))
    try:
        assert len(fig.axes) == 4
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        assert [title.split(".", 1)[0] for title in titles] == [
            "A", "B", "C", "D"]
        assert "0.810" in titles[0]
        assert "0.720" in titles[1]
        assert "0.630" in titles[2]
        assert "0.637" in titles[3]
        assert fig.axes[0].get_ylim() == pytest.approx((0.5, 1.0))
        assert fig.axes[1].get_ylim() == pytest.approx((0.5, 1.0))
    finally:
        plt.close(fig)


def test_rejects_non_test_report(tmp_path):
    reports = _reports(tmp_path)
    path = reports / "icu_readmission" / "overall.json"
    report = json.loads(path.read_text())
    report["metadata"]["split"] = "train"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="metadata.split must be 'test'"):
        plotting.build_auroc_figure(reports)


def test_saves_png(tmp_path):
    reports = _reports(tmp_path / "reports")
    output = plotting.save_auroc_figure(reports, tmp_path / "figure.png")

    assert output.is_file()
    assert output.stat().st_size > 0
