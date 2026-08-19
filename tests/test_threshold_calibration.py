import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/evaluation')))
from threshold_calibration import calibrate_binary_threshold, select_threshold_for_fold


def _make_separable_fold(n_negatives=200, n_positives=50, seed=0):
    """
    y_pred_probs cleanly separated by class (positives centered high,
    negatives centered low) so the target sensitivity is always reachable
    and the expected direction of the effect (lower target -> higher
    threshold -> higher specificity) is unambiguous.
    """
    rng = np.random.default_rng(seed)
    neg_probs = rng.uniform(0.0, 0.5, size=n_negatives)
    pos_probs = rng.uniform(0.5, 1.0, size=n_positives)
    y_true = np.concatenate([np.zeros(n_negatives), np.ones(n_positives)])
    y_pred_probs = np.concatenate([neg_probs, pos_probs])
    return y_true, y_pred_probs


def test_select_threshold_meets_target_sensitivity_when_reachable():
    y_true, y_pred_probs = _make_separable_fold()
    result = select_threshold_for_fold(y_true, y_pred_probs, target_sensitivity=0.90)

    assert result["target_met"] is True
    assert result["sensitivity"] >= 0.90
    assert 0.0 <= result["threshold"] <= 1.0


def test_select_threshold_prefers_higher_threshold_among_qualifying_ones():
    # Two positives at prob 0.6 and 0.9; at target_sensitivity=0.5 (>=1 of 2 positives)
    # both thresholds 0.6 and 0.9 satisfy sensitivity>=0.5, so the higher (0.9,
    # more specific) must win over the lower one.
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred_probs = np.array([0.1, 0.3, 0.5, 0.6, 0.9])

    result = select_threshold_for_fold(y_true, y_pred_probs, target_sensitivity=0.5)

    assert result["target_met"] is True
    assert result["threshold"] == 0.9
    assert result["sensitivity"] == 0.5


def test_select_threshold_falls_back_when_target_unreachable():
    # Only 1 positive, and even at its own probability, sensitivity tops out at 1.0.
    # An unreachable target (>1.0) can never be met - falls back to max achievable sensitivity.
    y_true = np.array([0, 0, 1])
    y_pred_probs = np.array([0.1, 0.2, 0.9])

    result = select_threshold_for_fold(y_true, y_pred_probs, target_sensitivity=1.5)

    assert result["target_met"] is False
    assert result["sensitivity"] == 1.0


def test_calibrate_binary_threshold_aggregates_via_median():
    fold_a_true, fold_a_probs = _make_separable_fold(seed=1)
    fold_b_true, fold_b_probs = _make_separable_fold(seed=2)
    fold_c_true, fold_c_probs = _make_separable_fold(seed=3)

    fold_results = [
        {"fold": 0, "y_true": fold_a_true, "y_pred_probs": fold_a_probs},
        {"fold": 1, "y_true": fold_b_true, "y_pred_probs": fold_b_probs},
        {"fold": 2, "y_true": fold_c_true, "y_pred_probs": fold_c_probs},
    ]

    result = calibrate_binary_threshold(fold_results, target_sensitivity=0.90)

    assert len(result["per_fold"]) == 3
    per_fold_thresholds = sorted(r["threshold"] for r in result["per_fold"])
    assert result["recommended_threshold"] == per_fold_thresholds[1]  # median of 3
    assert 0.0 <= result["recommended_threshold"] <= 1.0
    assert "sensitivity" in result["aggregate_at_recommended_threshold"]


def test_calibrate_binary_threshold_lower_target_yields_higher_or_equal_threshold():
    y_true, y_pred_probs = _make_separable_fold(seed=4)
    fold_results = [{"fold": 0, "y_true": y_true, "y_pred_probs": y_pred_probs}]

    strict = calibrate_binary_threshold(fold_results, target_sensitivity=0.99)
    lenient = calibrate_binary_threshold(fold_results, target_sensitivity=0.60)

    # Requiring lower sensitivity should allow an equal-or-more-specific (higher) threshold.
    assert lenient["recommended_threshold"] >= strict["recommended_threshold"]
