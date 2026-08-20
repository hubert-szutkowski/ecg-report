import numpy as np
from sklearn.metrics import classification_report


def macro_f1_with_support_floor(report: dict, class_names: list, min_support: int = 30) -> dict:
    """
    Recomputes macro F1 restricted to classes with support >= min_support, so a class measured
    on a handful of held-out examples (support-driven noise, not model quality) can't
    single-handedly swing the headline metric to near-zero. Classes below the floor are
    excluded and reported separately rather than silently dropped, so the reader can see what
    wasn't actually measured this run.

    Parameters:
        - report: sklearn classification_report(..., output_dict=True) dict
        - class_names: which class keys in report to consider
        - min_support: minimum support for a class to count toward the average
    Returns:
        - dict: min_support, macro_f1_floor, included_classes, excluded_classes
          (excluded_classes is a list of {"class": name, "support": int})
    """
    included, excluded, f1_values = [], [], []
    for name in class_names:
        if name not in report:
            continue
        support = int(report[name]["support"])
        if support >= min_support:
            included.append(name)
            f1_values.append(report[name]["f1-score"])
        else:
            excluded.append({"class": name, "support": support})

    return {
        "min_support": min_support,
        "macro_f1_floor": float(np.mean(f1_values)) if f1_values else float("nan"),
        "included_classes": included,
        "excluded_classes": excluded,
    }


def bootstrap_macro_f1_ci(
    y_true, y_pred, group_ids, class_names: list, min_support: int | None = None,
    n_bootstrap: int = 500, ci: float = 0.95, random_state: int = 42,
) -> dict:
    """
    Percentile bootstrap confidence interval for macro F1, resampling whole groups (patients /
    records), not individual beats, with replacement. Beats from the same patient are
    correlated (shared morphology, shared detector/model behavior on that recording), so
    resampling beats directly would understate the true uncertainty - the effective sample
    size is closer to the number of patients than the number of beats.

    Parameters:
        - y_true, y_pred: 1D arrays of class labels, one entry per beat
        - group_ids: 1D array aligned with y_true/y_pred, the patient/record id per beat
        - class_names: which classes classification_report should score
        - min_support: if given, each resample's macro F1 is restricted to classes with
          support >= this within THAT resample (a class can drop below the floor in some
          resamples even if it's above it in the full data - that's part of the uncertainty
          being measured, not an error)
        - n_bootstrap: number of resamples
        - ci: confidence level (e.g. 0.95 for a 95% interval)
        - random_state: seed for reproducibility
    Returns:
        - dict: point_estimate (from the original, non-resampled data), ci_low, ci_high,
          ci_level, n_bootstrap, n_groups
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_ids = np.asarray(group_ids)
    unique_groups = np.unique(group_ids)
    indices_by_group = {g: np.where(group_ids == g)[0] for g in unique_groups}

    def _macro_f1(true_arr, pred_arr):
        report = classification_report(true_arr, pred_arr, labels=class_names, output_dict=True, zero_division=0)
        if min_support is not None:
            return macro_f1_with_support_floor(report, class_names, min_support)["macro_f1_floor"]
        return report["macro avg"]["f1-score"]

    point_estimate = _macro_f1(y_true, y_pred)

    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([indices_by_group[g] for g in sampled_groups])
        score = _macro_f1(y_true[idx], y_pred[idx])
        if not np.isnan(score):
            scores.append(score)

    alpha = (1 - ci) / 2
    scores = np.array(scores)
    return {
        "point_estimate": point_estimate,
        "ci_low": float(np.percentile(scores, alpha * 100)) if len(scores) else float("nan"),
        "ci_high": float(np.percentile(scores, (1 - alpha) * 100)) if len(scores) else float("nan"),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_groups": len(unique_groups),
    }


def pick_best_cascade_threshold(candidate_results: list, class_names: list, min_support: int | None = None) -> dict:
    """
    Given full-cascade evaluation results for a set of candidate binary decision thresholds,
    picks whichever threshold maximizes macro F1 - directly on the metric that matters
    (the real cascade output), not a Stage-1-only proxy like sensitivity. Measured twice
    (see docs/next_steps_prompt.md) that optimizing a Stage 1 sensitivity target in isolation
    can *hurt* cascade_macro_f1 by flooding Stage 2 with false positives, which is why this
    replaces that approach rather than complementing it.

    Parameters:
        - candidate_results: list of {"threshold": float, "report": dict}, where "report" is a
          sklearn classification_report(output_dict=True) from running the FULL cascade
          (Pan-Tompkins -> Stage 1 -> Stage 2) at that threshold - not a Stage-1-only report
        - class_names: which classes to average over
        - min_support: if given, ranks candidates by macro_f1_floor (support >= min_support)
          instead of raw macro F1, so a class with near-zero held-out support in a given run
          can't decide the winner by noise
    Returns:
        - dict: candidates (per-threshold macro_f1/macro_f1_floor table), ranking_metric,
          min_support, best_threshold, best_macro_f1, best_macro_f1_floor
    """
    candidates = []
    for entry in candidate_results:
        report = entry["report"]
        macro_f1 = report["macro avg"]["f1-score"]
        floor_info = macro_f1_with_support_floor(report, class_names, min_support) if min_support is not None else None
        candidates.append({
            "threshold": entry["threshold"],
            "macro_f1": macro_f1,
            "macro_f1_floor": floor_info["macro_f1_floor"] if floor_info is not None else macro_f1,
        })

    ranking_key = "macro_f1_floor" if min_support is not None else "macro_f1"
    best = max(candidates, key=lambda c: c[ranking_key])

    return {
        "candidates": candidates,
        "ranking_metric": ranking_key,
        "min_support": min_support,
        "best_threshold": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "best_macro_f1_floor": best["macro_f1_floor"],
    }
