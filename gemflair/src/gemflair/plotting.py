import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PANELS = (
    ("A", "icu_daily_mortality", "continuous", "ICU daily mortality"),
    ("B", "icu_daily_ltach", "continuous", "ICU daily LTACH"),
    ("C", "extubation_failure_24h", "episodic",
     "Extubation failure within 24 hours"),
    ("D", "icu_readmission", "episodic", "Unplanned ICU readmission"),
)


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _ci(value, field):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain two values")
    low, high = (_number(v, field) for v in value)
    if not 0 <= low <= high <= 1:
        raise ValueError(f"{field} must be ordered within [0, 1]")
    return low, high


def _format_auc(value):
    return format(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP), "f")


def _load(reports_dir, task, mode):
    name = "landmark.json" if mode == "continuous" else "overall.json"
    path = Path(reports_dir) / task / name
    report = json.loads(path.read_text())
    metadata = report.get("metadata", {})
    expected = {"task": task, "report_mode": mode, "split": "test"}
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"{path}: metadata.{field} must be {value!r}, "
                f"got {metadata.get(field)!r}")
    return report


def _plot_continuous(ax, report, panel, title):
    pooled = report.get("pooled", {})
    pooled_auc = _number(pooled.get("auroc"), "pooled.auroc")
    pooled_ci = _ci(pooled.get("auroc_ci"), "pooled.auroc_ci")
    landmarks = report.get("by_landmark")
    if not isinstance(landmarks, list) or not landmarks:
        raise ValueError("by_landmark must be a non-empty list")

    days = np.array([
        _number(row.get("hospitalization_time"), "hospitalization_time")
        for row in landmarks
    ])
    auc = np.array([_number(row.get("auroc"), "by_landmark.auroc")
                    for row in landmarks])
    ci = np.array([_ci(row.get("auroc_ci"), "by_landmark.auroc_ci")
                   for row in landmarks])

    ax.fill_between(days, ci[:, 0], ci[:, 1], color="#1f77b4", alpha=0.18,
                    linewidth=0)
    ax.plot(days, auc, color="#1f77b4", marker="o", markersize=3,
            linewidth=1.6, label="Daily AUROC (95% CI)")
    ax.axhline(pooled_auc, color="#d62728", linestyle="--", linewidth=1.4,
               label="Pooled dynamic AUROC")
    ax.set(xlabel="Hospitalization day", ylabel="Time-specific AUROC",
           xlim=(days.min(), days.max()), ylim=(0.5, 1.0))
    ax.set_title(
        f"{panel}. {title}\nPooled dynamic AUROC {_format_auc(pooled_auc)} "
        f"(95% CI {_format_auc(pooled_ci[0])}-{_format_auc(pooled_ci[1])})",
        loc="left", fontsize=11, fontweight="semibold")
    ax.legend(loc="lower left", frameon=False, fontsize=8)


def _plot_episodic(ax, report, panel, title):
    discrimination = report.get("discrimination", {})
    auc = _number(discrimination.get("auroc"), "discrimination.auroc")
    ci = _ci(discrimination.get("auroc_ci"), "discrimination.auroc_ci")
    tpr = discrimination.get("tpr_at_grid")
    if not isinstance(tpr, list) or len(tpr) < 2:
        raise ValueError("discrimination.tpr_at_grid must contain at least two values")
    tpr = np.array([_number(v, "discrimination.tpr_at_grid") for v in tpr])
    if np.any((tpr < 0) | (tpr > 1)):
        raise ValueError("discrimination.tpr_at_grid must be within [0, 1]")
    fpr = np.linspace(0, 1, len(tpr))

    ax.plot(fpr, tpr, color="#1f77b4", linewidth=1.8, label="ROC curve")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.2,
            label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{panel}. {title}\nAUROC {_format_auc(auc)} "
        f"(95% CI {_format_auc(ci[0])}-{_format_auc(ci[1])})",
        loc="left", fontsize=11, fontweight="semibold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def build_auroc_figure(reports_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for ax, (panel, task, mode, title) in zip(axes.flat, PANELS):
        report = _load(reports_dir, task, mode)
        if mode == "continuous":
            _plot_continuous(ax, report, panel, title)
        else:
            _plot_episodic(ax, report, panel, title)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Discrimination on the held-out test split",
                 fontsize=14, fontweight="semibold")
    return fig


def save_auroc_figure(reports_dir, output):
    output = Path(output)
    fig = build_auroc_figure(reports_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output
