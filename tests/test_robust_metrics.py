import os
import sys

import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/evaluation')))
from robust_metrics import bootstrap_macro_f1_ci, macro_f1_with_support_floor, pick_best_cascade_threshold


def _report(y_true, y_pred, labels):
    return classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)


def test_macro_f1_with_support_floor_excludes_low_support_classes():
    # N and V well-supported and perfectly predicted; Q has 3 examples, all wrong (predicted as
    # "X", a class outside the ones we score, so it doesn't contaminate N/V's precision).
    y_true = ["N"] * 50 + ["V"] * 50 + ["Q"] * 3
    y_pred = ["N"] * 50 + ["V"] * 50 + ["X"] * 3
    report = _report(y_true, y_pred, labels=["N", "V", "Q"])

    result = macro_f1_with_support_floor(report, ["N", "V", "Q"], min_support=30)

    assert result["included_classes"] == ["N", "V"]
    assert result["excluded_classes"] == [{"class": "Q", "support": 3}]
    assert result["macro_f1_floor"] == 1.0  # N and V both perfect, Q correctly excluded from dragging it to 0


def test_macro_f1_with_support_floor_matches_plain_macro_when_all_classes_qualify():
    y_true = ["N"] * 40 + ["V"] * 40
    y_pred = ["N"] * 30 + ["V"] * 10 + ["V"] * 40
    report = _report(y_true, y_pred, labels=["N", "V"])
    plain_macro_f1 = report["macro avg"]["f1-score"]

    result = macro_f1_with_support_floor(report, ["N", "V"], min_support=30)

    assert result["excluded_classes"] == []
    assert abs(result["macro_f1_floor"] - plain_macro_f1) < 1e-9


def test_bootstrap_ci_brackets_point_estimate_for_stable_data():
    rng = np.random.default_rng(0)
    n_groups = 40
    y_true, y_pred, groups = [], [], []
    for g in range(n_groups):
        n = 20
        true_g = rng.choice(["N", "V"], size=n, p=[0.8, 0.2])
        # Model agrees with truth 90% of the time, independent of group.
        pred_g = np.where(rng.random(n) < 0.9, true_g, np.where(true_g == "N", "V", "N"))
        y_true.extend(true_g)
        y_pred.extend(pred_g)
        groups.extend([f"patient_{g}"] * n)

    result = bootstrap_macro_f1_ci(y_true, y_pred, groups, ["N", "V"], n_bootstrap=200, random_state=1)

    assert result["n_groups"] == n_groups
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    assert 0.0 <= result["ci_low"] <= result["ci_high"] <= 1.0


def test_bootstrap_ci_is_wider_with_fewer_groups():
    # Same total beats, but fewer, larger groups should carry more resampling uncertainty
    # (in the extreme, 1 group means every resample is either "all data" or impossible).
    rng = np.random.default_rng(2)

    def _make(n_groups, beats_per_group):
        y_true, y_pred, groups = [], [], []
        for g in range(n_groups):
            true_g = rng.choice(["N", "V"], size=beats_per_group, p=[0.7, 0.3])
            pred_g = np.where(rng.random(beats_per_group) < 0.85, true_g, np.where(true_g == "N", "V", "N"))
            y_true.extend(true_g)
            y_pred.extend(pred_g)
            groups.extend([f"g{g}"] * beats_per_group)
        return y_true, y_pred, groups

    y_true_many, y_pred_many, groups_many = _make(n_groups=50, beats_per_group=20)
    y_true_few, y_pred_few, groups_few = _make(n_groups=5, beats_per_group=200)

    result_many = bootstrap_macro_f1_ci(y_true_many, y_pred_many, groups_many, ["N", "V"], n_bootstrap=300, random_state=3)
    result_few = bootstrap_macro_f1_ci(y_true_few, y_pred_few, groups_few, ["N", "V"], n_bootstrap=300, random_state=3)

    width_many = result_many["ci_high"] - result_many["ci_low"]
    width_few = result_few["ci_high"] - result_few["ci_low"]
    assert width_few > width_many


def test_pick_best_cascade_threshold_selects_highest_macro_f1():
    reports = {
        0.3: _report(["N", "V", "V"], ["V", "V", "V"], labels=["N", "V"]),   # worse
        0.5: _report(["N", "V", "V"], ["N", "V", "N"], labels=["N", "V"]),   # mixed
        0.7: _report(["N", "V", "V"], ["N", "V", "V"], labels=["N", "V"]),   # perfect
    }
    candidate_results = [{"threshold": t, "report": r} for t, r in reports.items()]

    result = pick_best_cascade_threshold(candidate_results, ["N", "V"])

    assert result["best_threshold"] == 0.7
    assert result["best_macro_f1"] == 1.0
    assert len(result["candidates"]) == 3


def test_pick_best_cascade_threshold_ranks_by_floor_filtered_when_min_support_given():
    # At threshold A, a tiny class (support=2) is perfect, dragging naive macro F1 up despite
    # poor performance on the well-supported class - floor filtering should prefer threshold B.
    y_true_a = ["N"] * 40 + ["Q"] * 2
    y_pred_a = ["V"] * 40 + ["Q"] * 2  # N/V confusion is total, Q perfect
    y_true_b = ["N"] * 40 + ["Q"] * 2
    y_pred_b = ["N"] * 40 + ["N"] * 2  # N perfect, Q wrong (but Q has only 2 examples)

    candidate_results = [
        {"threshold": 0.3, "report": _report(y_true_a, y_pred_a, labels=["N", "Q"])},
        {"threshold": 0.6, "report": _report(y_true_b, y_pred_b, labels=["N", "Q"])},
    ]

    result = pick_best_cascade_threshold(candidate_results, ["N", "Q"], min_support=30)

    assert result["ranking_metric"] == "macro_f1_floor"
    assert result["best_threshold"] == 0.6
