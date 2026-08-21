import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
    }


def select_threshold_for_fold(y_true: np.ndarray, y_pred_probs: np.ndarray, target_sensitivity: float = 0.90) -> dict:
    """
    Picks the most specific decision threshold that still achieves at least
    target_sensitivity on one fold's held-out validation predictions - no
    retraining, just a different cutoff on probabilities the model already
    produces.

    Among all thresholds whose sensitivity (recall on the anomaly class)
    meets target_sensitivity, the highest such threshold is chosen (most
    specific / fewest false positives among the ones that still qualify).
    If no threshold reaches target_sensitivity on this fold (can happen with
    very few positives in a small validation partition), falls back to
    whichever threshold achieves the highest sensitivity possible and flags
    target_met=False so the caller can see it wasn't actually satisfied.

    Parameters:
        - y_true: np.ndarray of 0/1 ground truth labels
        - y_pred_probs: np.ndarray of predicted anomaly probabilities, same length as y_true
        - target_sensitivity: minimum acceptable sensitivity (0-1)
    Returns:
        - dict: threshold, target_met, sensitivity, specificity, ppv, npv
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    # roc_curve prepends a synthetic threshold (max_prob + 1) representing "predict nothing
    # positive" (fpr=tpr=0); it can never satisfy a positive target_sensitivity, so drop it -
    # otherwise it can spuriously win the argmax fallback below on a fold with zero positives.
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    meets_target = tpr >= target_sensitivity
    if meets_target.any():
        candidate_idx = np.where(meets_target)[0]
        chosen_idx = candidate_idx[np.argmax(thresholds[candidate_idx])]
        target_met = True
    else:
        chosen_idx = int(np.argmax(tpr))
        target_met = False

    chosen_threshold = float(thresholds[chosen_idx])
    y_pred = (y_pred_probs >= chosen_threshold).astype(int)
    return {"threshold": chosen_threshold, "target_met": target_met, **_confusion_metrics(y_true, y_pred)}


def calibrate_binary_threshold(fold_results: list, target_sensitivity: float = 0.90) -> dict:
    """
    Aggregates per-fold threshold selection into a single recommended
    threshold (median across folds), and reports what that recommended
    threshold achieves when applied uniformly to all folds' pooled
    validation predictions - the number that should be compared against the
    existing binary_clinical (fixed 0.5) section.

    Parameters:
        - fold_results: list of {"fold": int, "y_true": np.ndarray, "y_pred_probs": np.ndarray},
          one entry per CV fold (e.g. the binary_fold_results already computed by
          evaluate_binary_fold, which now also carries y_true/y_pred_probs)
        - target_sensitivity: minimum acceptable sensitivity passed to select_threshold_for_fold
    Returns:
        - dict: target_sensitivity, per_fold (list of per-fold selections),
          recommended_threshold (median), aggregate_at_recommended_threshold
    """
    per_fold = []
    for r in fold_results:
        selected = select_threshold_for_fold(r["y_true"], r["y_pred_probs"], target_sensitivity)
        per_fold.append({"fold": r["fold"], **selected})

    recommended_threshold = float(np.median([r["threshold"] for r in per_fold]))

    all_true = np.concatenate([r["y_true"] for r in fold_results])
    all_probs = np.concatenate([r["y_pred_probs"] for r in fold_results])
    y_pred_at_recommended = (all_probs >= recommended_threshold).astype(int)

    return {
        "target_sensitivity": target_sensitivity,
        "per_fold": per_fold,
        "recommended_threshold": recommended_threshold,
        "aggregate_at_recommended_threshold": _confusion_metrics(all_true, y_pred_at_recommended),
    }
