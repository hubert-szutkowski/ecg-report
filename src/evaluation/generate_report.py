import os
import sys
import glob
import json
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import wfdb
import tensorflow as tf
import mlflow

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
MULTISTAGE_DIR = os.path.join(SRC_DIR, "AAMI_classification_multistages")
if MULTISTAGE_DIR not in sys.path:
    sys.path.insert(0, MULTISTAGE_DIR)

import cascade
from pan_tompkins import pan_tompkins_detect
from multistage_preprocessing import get_record_ids, SYMBOL_TO_CLASS
from multistage_train import load_data
from multistage_model import PositionalEmbedding

CLINICAL_CLASS_ORDER = ["N", "S", "V", "F", "Q"]
COMPUTE_SKU = "Standard_DS3_v2"  # matches Azure/submit_job.py's cpu-cluster instance size
CUSTOM_OBJECTS = {"PositionalEmbedding": PositionalEmbedding}


def build_parser():
    parser = argparse.ArgumentParser(description="ECG Cascade Evaluation Report")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw ECG data")
    parser.add_argument("--train-job-name", type=str, required=True, help="Azure ML job name (= mlflow run id) whose outputs to evaluate")
    parser.add_argument("--train-outputs-dir", type=str, required=True, help="Mounted directory of --train-job-name's outputs")
    parser.add_argument("--train-job-name-nogan", type=str, default=None, help="Optional no-GAN baseline job name, for the GAN-vs-no-GAN comparison")
    parser.add_argument("--train-outputs-nogan-dir", type=str, default=None, help="Mounted directory of --train-job-name-nogan's outputs")
    parser.add_argument("--fold", type=int, default=None, help="Restrict per-fold sections to a single fold")
    parser.add_argument("--records", type=str, default=None, help="Comma-separated record ids for the Pan-Tompkins/CPU benchmark sections (default: first 10)")
    parser.add_argument("--champion-model-name", type=str, default="ecg_multiclass_model", help="Registered model name to compare against")
    parser.add_argument("--metric-name", type=str, default="cascade_macro_f1", help="Metric that drives the promotion decision")
    parser.add_argument("--tolerance-samples", type=int, default=5, help="Peak-matching tolerance in samples, shared by 1.1 and 1.9")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to write the report/metrics/figures")
    return parser


def load_training_run_params(train_job_name: str) -> dict:
    run = mlflow.get_run(train_job_name)
    return {
        "selected_samples": int(run.data.params["selected_samples"]),
        "random_seed": int(run.data.params["random_seed"]),
        "Folds": int(run.data.params["Folds"]),
    }


def load_fold_splits(train_outputs_dir: str, stage: str) -> dict:
    path = os.path.join(train_outputs_dir, f"fold_splits_{stage}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The training job's outputs don't include fold_splits_{stage}.json - "
            "it was likely run with an older multistage_train.py that predates fold-split persistence. "
            "Re-run Azure/submit_job.py to produce a training job with current code, then evaluate that one."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_label_encoder(train_outputs_dir: str):
    return joblib.load(os.path.join(train_outputs_dir, "label_encoder.pkl"))


def load_fold_artifacts(train_outputs_dir: str, stage: str, fold_number: int):
    suffix = "_binary" if stage == "binary" else ""
    model_path = os.path.join(train_outputs_dir, f"best_model_fold_{fold_number}{suffix}.keras")
    scaler_path = os.path.join(train_outputs_dir, f"scaler_fold_{fold_number}{suffix}.pkl")
    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_full_dataset(data_dir: str, stage: str, run_params: dict, expected_window_size: int, encoder=None):
    X_data, y_labels, groups, window_size = load_data(
        data_dir, run_params["selected_samples"], run_params["random_seed"], stage
    )
    if window_size != expected_window_size:
        raise ValueError(
            f"Reconstructed window_size ({window_size}) for stage='{stage}' does not match "
            f"the training run's persisted window_size ({expected_window_size}) - "
            "data_dir/selected_samples/random_seed must have drifted from the training run."
        )
    X_data = X_data.astype(np.float32)
    y_encoded = y_labels.astype(np.int32) if stage == "binary" else encoder.transform(y_labels)
    return X_data, y_encoded, groups


def reconstruct_partition(X_data, y_encoded, groups, record_ids):
    mask = np.isin(groups, list(record_ids))
    return X_data[mask], y_encoded[mask], groups[mask]


def predict_fold(model, scaler, X_raw, window_size):
    X_scaled = scaler.transform(X_raw).reshape(-1, window_size, 1)
    return model.predict(X_scaled, verbose=0)


def resolve_record_ids(data_dir: str, records_arg: str | None) -> list:
    all_ids = get_record_ids(data_dir)
    if not records_arg:
        return all_ids[:10]
    requested = [r.strip() for r in records_arg.split(",") if r.strip()]
    missing = [r for r in requested if r not in all_ids]
    if missing:
        raise ValueError(f"Requested records not found in data_dir: {missing}")
    return requested


def restrict_to_fold(splits: dict, fold: int | None) -> dict:
    if fold is None:
        return splits
    matching = [s for s in splits["folds"] if s["fold"] == fold]
    if not matching:
        raise ValueError(f"Fold {fold} not found in persisted fold splits")
    return {"window_size": splits["window_size"], "folds": matching}


def _markdown_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


# --- Per-fold evaluation ---

def evaluate_multiclass_fold(train_outputs_dir, X_data, y_encoded, groups, fold_split, encoder, num_classes, window_size):
    fold = fold_split["fold"]
    model, scaler = load_fold_artifacts(train_outputs_dir, "multiclass", fold)
    X_val, y_val, _ = reconstruct_partition(X_data, y_encoded, groups, fold_split["val_record_ids"])

    y_pred_probs = predict_fold(model, scaler, X_val, window_size)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_val, y_pred, labels=range(num_classes), target_names=list(encoder.classes_),
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_val, y_pred, labels=range(num_classes))

    try:
        present = np.unique(y_val)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)
        val_auc = roc_auc_score(y_val_cat[:, present], y_pred_probs[:, present], average="weighted", multi_class="ovr")
    except Exception:
        val_auc = float("nan")

    return {
        "fold": fold, "y_true": y_val, "y_pred": y_pred, "report": report, "confusion_matrix": cm,
        "val_auc": val_auc,
    }


def evaluate_binary_fold(train_outputs_dir, X_data, y_encoded, groups, fold_split, window_size):
    fold = fold_split["fold"]
    model, scaler = load_fold_artifacts(train_outputs_dir, "binary", fold)
    X_val, y_val, _ = reconstruct_partition(X_data, y_encoded, groups, fold_split["val_record_ids"])

    y_pred_probs = predict_fold(model, scaler, X_val, window_size).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    return {
        "fold": fold, "confusion_matrix": cm, "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv,
    }


# --- 1.7 dataset composition ---

def section_dataset_composition(binary_splits, multiclass_splits, X_multi, y_multi, groups_multi, encoder):
    rows = []
    for split in multiclass_splits["folds"]:
        _, y_train, _ = reconstruct_partition(X_multi, y_multi, groups_multi, split["train_record_ids"])
        _, y_val, _ = reconstruct_partition(X_multi, y_multi, groups_multi, split["val_record_ids"])
        train_counts = {name: int(np.sum(y_train == idx)) for idx, name in enumerate(encoder.classes_)}
        val_counts = {name: int(np.sum(y_val == idx)) for idx, name in enumerate(encoder.classes_)}
        rows.append({
            "fold": split["fold"],
            "train_patients": len(split["train_record_ids"]),
            "val_patients": len(split["val_record_ids"]),
            "train_class_counts": train_counts,
            "val_class_counts": val_counts,
        })
    return {"binary_folds": binary_splits["folds"], "multiclass_folds": rows}


# --- 1.3 per-fold table ---

def section_per_fold_table(train_outputs_dir, multiclass_fold_results):
    metrics_df = pd.read_csv(os.path.join(train_outputs_dir, "metrics.csv"))
    rows = []
    for result in multiclass_fold_results:
        fold = result["fold"]
        csv_row = metrics_df[metrics_df["Fold"] == fold].iloc[0]
        weighted_f1 = result["report"]["weighted avg"]["f1-score"]
        macro_f1 = result["report"]["macro avg"]["f1-score"]
        support = {name: int(result["report"][name]["support"]) for name in result["report"] if name in CLINICAL_CLASS_ORDER}
        rows.append({
            "fold": fold, "val_acc": float(csv_row["Val Acc"]), "val_auc": float(result["val_auc"]),
            "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1), "support": support,
        })
    return rows


# --- 1.2 aggregate multiclass report + confusion matrix ---

def section_aggregate_multiclass(multiclass_fold_results, encoder, num_classes):
    all_true = np.concatenate([r["y_true"] for r in multiclass_fold_results])
    all_pred = np.concatenate([r["y_pred"] for r in multiclass_fold_results])
    report = classification_report(
        all_true, all_pred, labels=range(num_classes), target_names=list(encoder.classes_),
        output_dict=True, zero_division=0,
    )
    cm = sum(r["confusion_matrix"] for r in multiclass_fold_results)
    return {"report": report, "confusion_matrix": cm}


def reorder_confusion_matrix(cm, current_labels, target_labels):
    target_labels = [lbl for lbl in target_labels if lbl in current_labels]
    idx = [list(current_labels).index(lbl) for lbl in target_labels]
    return cm[np.ix_(idx, idx)], target_labels


# --- 1.4 binary clinical metrics ---

def section_binary_clinical_metrics(binary_fold_results):
    total_cm = sum(r["confusion_matrix"] for r in binary_fold_results)
    tn, fp, fn, tp = total_cm.ravel()
    aggregate = {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
    }
    per_fold = [
        {"fold": r["fold"], "sensitivity": r["sensitivity"], "specificity": r["specificity"], "ppv": r["ppv"], "npv": r["npv"]}
        for r in binary_fold_results
    ]
    return {"aggregate": aggregate, "per_fold": per_fold}


# --- 1.5 baselines ---

def compute_baseline(X_train, y_train, X_val, y_val, random_seed, labels, target_names):
    results = {}
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    for strategy in ("most_frequent", "stratified"):
        clf = DummyClassifier(strategy=strategy, random_state=random_seed)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_val_flat)
        results[strategy] = classification_report(
            y_val, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
        )
    return results


def section_baselines(X_bin, y_bin, groups_bin, binary_splits, X_multi, y_multi, groups_multi, multiclass_splits, encoder, num_classes, random_seed):
    binary_rows = []
    for split in binary_splits["folds"]:
        X_train, y_train, _ = reconstruct_partition(X_bin, y_bin, groups_bin, split["train_record_ids"])
        X_val, y_val, _ = reconstruct_partition(X_bin, y_bin, groups_bin, split["val_record_ids"])
        baseline = compute_baseline(X_train, y_train, X_val, y_val, random_seed, labels=[0, 1], target_names=["N", "anomaly"])
        binary_rows.append({"fold": split["fold"], **{k: v["macro avg"]["f1-score"] for k, v in baseline.items()}})

    multiclass_rows = []
    for split in multiclass_splits["folds"]:
        X_train, y_train, _ = reconstruct_partition(X_multi, y_multi, groups_multi, split["train_record_ids"])
        X_val, y_val, _ = reconstruct_partition(X_multi, y_multi, groups_multi, split["val_record_ids"])
        baseline = compute_baseline(
            X_train, y_train, X_val, y_val, random_seed,
            labels=range(num_classes), target_names=list(encoder.classes_),
        )
        multiclass_rows.append({"fold": split["fold"], **{k: v["macro avg"]["f1-score"] for k, v in baseline.items()}})

    return {"binary_macro_f1": binary_rows, "multiclass_macro_f1": multiclass_rows}


# --- 1.9 Pan-Tompkins detector validation ---

def evaluate_pan_tompkins(data_dir, record_ids, tolerance_samples):
    rows = []
    total_matched = total_annotated = total_detected = 0

    for record_id in record_ids:
        record_path = os.path.join(data_dir, record_id)
        ann = wfdb.rdann(record_path, "atr")
        symbols = np.array(ann.symbol)
        samples = np.array(ann.sample)
        known_mask = np.array([s in SYMBOL_TO_CLASS for s in symbols])
        annotated_samples = samples[known_mask]

        signal, fields = wfdb.rdsamp(record_path, channels=[0])
        fs = fields["fs"]
        detected = pan_tompkins_detect(signal[:, 0].astype(np.float32), fs)

        match = cascade.match_detected_to_annotated_peaks(detected, annotated_samples, tolerance_samples)
        n_matched = len(match["matched_annotated_idx"])
        n_annotated = len(annotated_samples)
        n_detected = len(detected)

        rows.append({
            "record": record_id, "n_annotated": n_annotated, "n_detected": n_detected, "n_matched": n_matched,
            "sensitivity": n_matched / n_annotated if n_annotated else float("nan"),
            "ppv": n_matched / n_detected if n_detected else float("nan"),
        })
        total_matched += n_matched
        total_annotated += n_annotated
        total_detected += n_detected

    aggregate = {
        "sensitivity": total_matched / total_annotated if total_annotated else float("nan"),
        "ppv": total_matched / total_detected if total_detected else float("nan"),
    }
    return rows, aggregate


# --- 1.1 end-to-end cascade ---

def section_cascade_end_to_end(data_dir, train_outputs_dir, binary_splits, multiclass_splits, encoder, window_size, tolerance_samples):
    all_true, all_pred = [], []
    loss_breakdown = {"missed_by_detector": 0, "classified_normal_by_stage1": 0, "misclassified_stage2": 0, "correctly_classified": 0}
    n_records_evaluated = 0

    multiclass_fold_by_number = {s["fold"]: s for s in multiclass_splits["folds"]}

    for bin_split in binary_splits["folds"]:
        fold = bin_split["fold"]
        multi_split = multiclass_fold_by_number.get(fold)
        if multi_split is None:
            continue
        record_ids = sorted(set(bin_split["val_record_ids"]) & set(multi_split["val_record_ids"]))
        if not record_ids:
            continue

        binary_model, binary_scaler = load_fold_artifacts(train_outputs_dir, "binary", fold)
        multiclass_model, _ = load_fold_artifacts(train_outputs_dir, "multiclass", fold)

        for record_id in record_ids:
            record_path = os.path.join(data_dir, record_id)
            ann = wfdb.rdann(record_path, "atr")
            symbols = np.array(ann.symbol)
            samples = np.array(ann.sample)
            aami_labels = np.array([SYMBOL_TO_CLASS.get(s) for s in symbols], dtype=object)
            known_mask = aami_labels != None  # noqa: E711
            annotated_samples = samples[known_mask]
            annotated_labels = aami_labels[known_mask]

            signal, fields = wfdb.rdsamp(record_path, channels=[0])
            fs = fields["fs"]

            try:
                cascade_result = cascade.run_cascade_batch(
                    signal[:, 0].astype(np.float32), fs, binary_model=binary_model, scaler=binary_scaler,
                    multiclass_model=multiclass_model, window_size=window_size,
                )
            except ValueError:
                continue

            n_records_evaluated += 1
            detected_peaks = np.array([beat["peak_sample"] for beat in cascade_result["results"]])
            match = cascade.match_detected_to_annotated_peaks(detected_peaks, annotated_samples, tolerance_samples)

            for a_idx in match["unmatched_annotated_idx"]:
                true_label = annotated_labels[a_idx]
                if true_label == "N":
                    continue
                loss_breakdown["missed_by_detector"] += 1
                all_true.append(true_label)
                all_pred.append("N")

            for a_idx, d_idx in zip(match["matched_annotated_idx"], match["matched_detected_idx"]):
                true_label = annotated_labels[a_idx]
                beat = cascade_result["results"][d_idx]
                if not beat["is_anomaly"]:
                    pred_label = "N"
                else:
                    pred_label = encoder.classes_[beat["predicted_class"]] if beat["predicted_class"] is not None else "N"

                all_true.append(true_label)
                all_pred.append(pred_label)

                if true_label == "N":
                    continue
                if pred_label == true_label:
                    loss_breakdown["correctly_classified"] += 1
                elif pred_label == "N":
                    loss_breakdown["classified_normal_by_stage1"] += 1
                else:
                    loss_breakdown["misclassified_stage2"] += 1

    labels_present = [lbl for lbl in CLINICAL_CLASS_ORDER if lbl in set(all_true) | set(all_pred)]
    report = classification_report(all_true, all_pred, labels=labels_present, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=labels_present)

    return {
        "report": report, "confusion_matrix": cm, "confusion_matrix_labels": labels_present,
        "loss_breakdown": loss_breakdown, "n_records_evaluated": n_records_evaluated,
    }


# --- 1.6 GAN vs no-GAN ---

def section_gan_vs_nogan(train_outputs_dir, train_outputs_nogan_dir):
    if not train_outputs_nogan_dir:
        return None

    gan_metrics = pd.read_csv(os.path.join(train_outputs_dir, "metrics.csv"))
    nogan_metrics = pd.read_csv(os.path.join(train_outputs_nogan_dir, "metrics.csv"))
    merged = gan_metrics.merge(nogan_metrics, on="Fold", suffixes=("_gan", "_nogan"))
    merged["val_acc_diff"] = merged["Val Acc_gan"] - merged["Val Acc_nogan"]
    merged["val_auc_diff"] = merged["Val AUC_gan"] - merged["Val AUC_nogan"]
    merged["f1_diff"] = merged["F1 Score_gan"] - merged["F1 Score_nogan"]

    fold_variance = float(gan_metrics["Val Acc"].std())
    mean_diff = float(merged["val_acc_diff"].mean())
    verdict = (
        "GAN augmentation's effect on Val Acc exceeds inter-fold variance"
        if abs(mean_diff) > fold_variance
        else "GAN augmentation's effect on Val Acc is within inter-fold variance - inconclusive"
    )
    return {"table": merged.to_dict(orient="records"), "mean_val_acc_diff": mean_diff, "fold_variance": fold_variance, "verdict": verdict}


# --- 1.8 CPU inference benchmarks ---

def section_cpu_benchmarks(data_dir, record_ids, binary_model, binary_scaler, multiclass_model, window_size, binary_model_path, multiclass_model_path):
    detector_times, stage1_times, stage2_times, per_beat_latencies = [], [], [], []
    per_window_single_times = []
    total_windows = 0

    for record_id in record_ids:
        record_path = os.path.join(data_dir, record_id)
        signal, fields = wfdb.rdsamp(record_path, channels=[0])
        fs = fields["fs"]
        signal = signal[:, 0].astype(np.float32)

        t0 = time.perf_counter()
        peaks = pan_tompkins_detect(signal, fs)
        t1 = time.perf_counter()
        detector_times.append(t1 - t0)

        windows_raw, valid_peaks = cascade.extract_windows_around_peaks(signal, peaks, window_size)
        if len(valid_peaks) == 0:
            continue

        windows_norm = cascade.normalize_windows(windows_raw, binary_scaler)
        x_all = windows_norm.reshape(-1, window_size, 1)

        t2 = time.perf_counter()
        binary_probs = binary_model.predict(x_all, batch_size=256, verbose=0).flatten()
        t3 = time.perf_counter()
        stage1_times.append(t3 - t2)

        anomaly_idx = np.where(binary_probs >= 0.5)[0]
        t4 = time.perf_counter()
        if len(anomaly_idx) > 0:
            multiclass_model.predict(x_all[anomaly_idx], batch_size=256, verbose=0)
        t5 = time.perf_counter()
        stage2_times.append(t5 - t4)

        total_windows += len(valid_peaks)
        record_total = (t1 - t0) + (t3 - t2) + (t5 - t4)
        per_beat_latencies.extend([record_total / len(valid_peaks)] * len(valid_peaks))

        sample_size = min(20, len(x_all))
        t6 = time.perf_counter()
        for i in range(sample_size):
            binary_model.predict(x_all[i:i + 1], batch_size=1, verbose=0)
        t7 = time.perf_counter()
        per_window_single_times.append((t7 - t6) / sample_size)

    batched_sec_per_window = float(np.sum(stage1_times) / total_windows) if total_windows else float("nan")
    per_window_sec = float(np.mean(per_window_single_times)) if per_window_single_times else float("nan")

    return {
        "n_records": len(record_ids),
        "wall_clock_detector_sec_mean": float(np.mean(detector_times)) if detector_times else float("nan"),
        "wall_clock_stage1_sec_mean": float(np.mean(stage1_times)) if stage1_times else float("nan"),
        "wall_clock_stage2_sec_mean": float(np.mean(stage2_times)) if stage2_times else float("nan"),
        "per_beat_latency_mean_ms": float(np.mean(per_beat_latencies) * 1000) if per_beat_latencies else float("nan"),
        "per_beat_latency_p50_ms": float(np.percentile(per_beat_latencies, 50) * 1000) if per_beat_latencies else float("nan"),
        "per_beat_latency_p95_ms": float(np.percentile(per_beat_latencies, 95) * 1000) if per_beat_latencies else float("nan"),
        "batched_predict_sec_per_window": batched_sec_per_window,
        "per_window_predict_sec_per_window": per_window_sec,
        "batching_speedup_x": (per_window_sec / batched_sec_per_window) if batched_sec_per_window else float("nan"),
        "binary_model_params": int(binary_model.count_params()),
        "multiclass_model_params": int(multiclass_model.count_params()) if multiclass_model is not None else None,
        "binary_model_size_bytes": os.path.getsize(binary_model_path),
        "multiclass_model_size_bytes": os.path.getsize(multiclass_model_path) if multiclass_model_path and os.path.exists(multiclass_model_path) else None,
        "compute_sku": COMPUTE_SKU,
    }


# --- Figures ---

def plot_confusion_matrix_figure(cm, labels, output_path, title):
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical", values_format=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_per_fold_metrics_figure(metrics_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df))
    width = 0.25
    ax.bar(x - width, metrics_df["Val Acc"], width, label="Val Acc")
    ax.bar(x, metrics_df["Val AUC"], width, label="Val AUC")
    ax.bar(x + width, metrics_df["F1 Score"], width, label="F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["Fold"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Per-fold multiclass metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_pan_tompkins_detection_figure(data_dir, record_id, output_path, duration_sec=6.0):
    record_path = os.path.join(data_dir, record_id)
    signal, fields = wfdb.rdsamp(record_path, channels=[0])
    fs = fields["fs"]
    signal = signal[:, 0].astype(np.float32)

    ann = wfdb.rdann(record_path, "atr")
    symbols = np.array(ann.symbol)
    samples = np.array(ann.sample)
    known_mask = np.array([s in SYMBOL_TO_CLASS for s in symbols])
    annotated_samples = samples[known_mask]

    peaks, debug = pan_tompkins_detect(signal, fs, return_intermediate=True)

    n_show = min(int(duration_sec * fs), len(signal))
    t = np.arange(n_show) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, signal[:n_show], color="black", linewidth=0.8, label="Raw ECG")
    shown_ann = annotated_samples[annotated_samples < n_show]
    axes[0].scatter(shown_ann / fs, signal[shown_ann], color="green", marker="o", label="Ground truth")
    shown_det = peaks[peaks < n_show]
    axes[0].scatter(shown_det / fs, signal[shown_det], color="red", marker="x", label="Detected")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Pan-Tompkins detection - record {record_id}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, debug["integrated"][:n_show], color="blue", label="Integrated signal")
    axes[1].axhline(debug["threshold"], color="orange", linestyle="--", label="Adaptive threshold")
    axes[1].set_ylabel("Energy")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_wgan_real_vs_synthetic_figure(train_outputs_dir, X_multi, y_multi, groups_multi, multiclass_splits, encoder, window_size, output_path):
    synthetic_files = sorted(glob.glob(os.path.join(train_outputs_dir, "gan_synthetic_*_fold_*.npy")))
    splits_by_fold = {s["fold"]: s for s in multiclass_splits["folds"]}

    chosen = None
    for path in synthetic_files:
        stem = os.path.basename(path)[len("gan_synthetic_"):-len(".npy")]
        class_name, fold_str = stem.rsplit("_fold_", 1)
        fold_number = int(fold_str)
        if fold_number in splits_by_fold:
            chosen = (path, class_name, fold_number)
            break

    if chosen is None:
        return False

    synthetic_path, class_name, fold_number = chosen
    synthetic = np.load(synthetic_path).reshape(-1, window_size)

    class_idx = list(encoder.classes_).index(class_name)
    fold_split = splits_by_fold[fold_number]
    train_mask = np.isin(groups_multi, fold_split["train_record_ids"]) & (y_multi == class_idx)
    real = X_multi[train_mask]

    n_overlay = min(8, len(real), len(synthetic))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i in range(n_overlay):
        axes[0].plot(real[i], color="green", alpha=0.6, linewidth=1, label="Real" if i == 0 else None)
        axes[0].plot(synthetic[i], color="blue", alpha=0.6, linewidth=1, label="Synthetic" if i == 0 else None)
    axes[0].set_title(f"Real vs synthetic beat overlay ({class_name}, fold {fold_number})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real.reshape(len(real), -1))
    synth_pca = pca.transform(synthetic.reshape(len(synthetic), -1))
    axes[1].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label="Real")
    axes[1].scatter(synth_pca[:, 0], synth_pca[:, 1], alpha=0.5, label="Synthetic")
    axes[1].set_title("PCA: real vs synthetic distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return True


# --- Promotion decision (auth: relies on the job's managed identity, see plan) ---

def get_workspace_ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )


def build_promotion_decision(challenger_metrics: dict, champion_model_name: str, metric_name: str) -> dict:
    reason_prefix = f"metric={metric_name}"
    challenger_score = challenger_metrics.get(metric_name)
    if challenger_score is None:
        return {
            "promote": False,
            "reason": f"{reason_prefix}: challenger did not produce this metric",
            "challenger_metrics": challenger_metrics,
            "champion_metrics": {},
        }

    try:
        ml_client = get_workspace_ml_client()
        champion_model = ml_client.models.get(name=champion_model_name, label="latest")
        champion_score = float(champion_model.tags.get(metric_name, "-inf"))
        champion_metrics = dict(champion_model.tags)
    except Exception as exc:
        return {
            "promote": True,
            "reason": f"{reason_prefix}: no champion found or workspace unreachable ({exc}) - promoting as first deployment",
            "challenger_metrics": challenger_metrics,
            "champion_metrics": {},
        }

    promote = challenger_score > champion_score
    reason = (
        f"{reason_prefix}: challenger {challenger_score:.4f} "
        f"{'beats' if promote else 'does not beat'} champion {champion_score:.4f}"
    )
    return {"promote": promote, "reason": reason, "challenger_metrics": challenger_metrics, "champion_metrics": champion_metrics}


# --- Report assembly ---

def build_markdown_report(sections: dict) -> str:
    lines = ["# ECG Cascade Evaluation Report", ""]

    lines += ["## 1.1 End-to-end cascade metrics", ""]
    cascade_section = sections["cascade"]
    lines.append(f"Records evaluated: {cascade_section['n_records_evaluated']}")
    lines.append("")
    report = cascade_section["report"]
    headers = ["class", "precision", "recall", "f1-score", "support"]
    rows = [[name, report[name]["precision"], report[name]["recall"], report[name]["f1-score"], int(report[name]["support"])]
            for name in cascade_section["confusion_matrix_labels"]]
    lines.append(_markdown_table(headers, rows))
    lines.append("")
    lines.append(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    lines.append("")
    lb = cascade_section["loss_breakdown"]
    lines.append("Where true anomalous beats are lost:")
    lines.append(_markdown_table(
        ["outcome", "count"],
        [["missed by R-peak detector", lb["missed_by_detector"]],
         ["classified Normal by stage 1", lb["classified_normal_by_stage1"]],
         ["misclassified at stage 2", lb["misclassified_stage2"]],
         ["correctly classified end-to-end", lb["correctly_classified"]]],
    ))
    lines.append("")

    lines += ["## 1.2 Multiclass classification report (aggregated across folds)", ""]
    agg = sections["aggregate_multiclass"]["report"]
    rows = [[name, agg[name]["precision"], agg[name]["recall"], agg[name]["f1-score"], int(agg[name]["support"])]
            for name in CLINICAL_CLASS_ORDER if name in agg]
    lines.append(_markdown_table(headers, rows))
    lines.append("")

    lines += ["## 1.3 Per-fold table (multiclass)", ""]
    per_fold = sections["per_fold_table"]
    lines.append(_markdown_table(
        ["fold", "val_acc", "val_auc", "macro_f1", "weighted_f1", "support (per class)"],
        [[r["fold"], r["val_acc"], r["val_auc"], r["macro_f1"], r["weighted_f1"],
          " ".join(f"{name}:{count}" for name, count in r["support"].items())] for r in per_fold],
    ))
    lines.append("")

    lines += ["## 1.4 Stage 1 clinical metrics", ""]
    clinical = sections["binary_clinical"]["aggregate"]
    lines.append(_markdown_table(
        ["metric", "value"],
        [["Sensitivity", clinical["sensitivity"]], ["Specificity", clinical["specificity"]],
         ["PPV", clinical["ppv"]], ["NPV", clinical["npv"]]],
    ))
    lines.append("")

    lines += ["## 1.5 Baselines (macro F1)", ""]
    baselines = sections["baselines"]
    lines.append("Binary:")
    lines.append(_markdown_table(
        ["fold", "most_frequent", "stratified"],
        [[r["fold"], r["most_frequent"], r["stratified"]] for r in baselines["binary_macro_f1"]],
    ))
    lines.append("")
    lines.append("Multiclass:")
    lines.append(_markdown_table(
        ["fold", "most_frequent", "stratified"],
        [[r["fold"], r["most_frequent"], r["stratified"]] for r in baselines["multiclass_macro_f1"]],
    ))
    lines.append("")

    lines += ["## 1.6 GAN vs. no-GAN augmentation", ""]
    gan_vs_nogan = sections["gan_vs_nogan"]
    if gan_vs_nogan is None:
        lines.append("No no-GAN baseline run available. To generate one, run `Azure/submit_job_nogan_baseline.py`.")
    else:
        lines.append(_markdown_table(
            ["fold", "val_acc_gan", "val_acc_nogan", "val_acc_diff"],
            [[r["Fold"], r["Val Acc_gan"], r["Val Acc_nogan"], r["val_acc_diff"]] for r in gan_vs_nogan["table"]],
        ))
        lines.append("")
        lines.append(f"Mean Val Acc difference: {gan_vs_nogan['mean_val_acc_diff']:.4f} (inter-fold std: {gan_vs_nogan['fold_variance']:.4f})")
        lines.append(f"Verdict: {gan_vs_nogan['verdict']}")
    lines.append("")

    lines += ["## 1.7 Dataset composition", ""]
    composition = sections["dataset_composition"]
    lines.append(_markdown_table(
        ["fold", "train_patients", "val_patients"],
        [[r["fold"], r["train_patients"], r["val_patients"]] for r in composition["multiclass_folds"]],
    ))
    lines.append("")

    lines += ["## 1.8 CPU inference benchmarks", ""]
    bench = sections["cpu_benchmarks"]
    lines.append(_markdown_table(
        ["metric", "value"],
        [["Records benchmarked", bench["n_records"]],
         ["Compute SKU", bench["compute_sku"]],
         ["Mean detector time (s/record)", bench["wall_clock_detector_sec_mean"]],
         ["Mean stage 1 time (s/record)", bench["wall_clock_stage1_sec_mean"]],
         ["Mean stage 2 time (s/record)", bench["wall_clock_stage2_sec_mean"]],
         ["Per-beat latency mean (ms)", bench["per_beat_latency_mean_ms"]],
         ["Per-beat latency p50 (ms)", bench["per_beat_latency_p50_ms"]],
         ["Per-beat latency p95 (ms)", bench["per_beat_latency_p95_ms"]],
         ["Batching speedup (x)", bench["batching_speedup_x"]],
         ["Binary model params", bench["binary_model_params"]],
         ["Multiclass model params", bench["multiclass_model_params"]],
         ["Binary model size (bytes)", bench["binary_model_size_bytes"]],
         ["Multiclass model size (bytes)", bench["multiclass_model_size_bytes"]]],
    ))
    lines.append("")

    lines += ["## 1.9 Pan-Tompkins detector validation", ""]
    pt_rows, pt_aggregate = sections["pan_tompkins"]
    lines.append(_markdown_table(
        ["record", "n_annotated", "n_detected", "n_matched", "sensitivity", "ppv"],
        [[r["record"], r["n_annotated"], r["n_detected"], r["n_matched"], r["sensitivity"], r["ppv"]] for r in pt_rows],
    ))
    lines.append("")
    lines.append(f"Aggregate sensitivity: {pt_aggregate['sensitivity']:.4f} | Aggregate PPV: {pt_aggregate['ppv']:.4f}")
    lines.append("")

    return "\n".join(lines)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    args = build_parser().parse_args()

    output_dir = args.output_dir
    img_dir = os.path.join(output_dir, "docs", "img")
    os.makedirs(img_dir, exist_ok=True)

    run_params = load_training_run_params(args.train_job_name)
    binary_splits = load_fold_splits(args.train_outputs_dir, "binary")
    multiclass_splits = load_fold_splits(args.train_outputs_dir, "multiclass")
    encoder = load_label_encoder(args.train_outputs_dir)
    num_classes = len(encoder.classes_)

    window_size = binary_splits["window_size"]
    if multiclass_splits["window_size"] != window_size:
        raise ValueError("Binary and multiclass fold splits disagree on window_size - training run is inconsistent")

    X_bin, y_bin, groups_bin = load_full_dataset(args.data_dir, "binary", run_params, window_size)
    X_multi, y_multi, groups_multi = load_full_dataset(args.data_dir, "multiclass", run_params, window_size, encoder=encoder)

    binary_splits = restrict_to_fold(binary_splits, args.fold)
    multiclass_splits = restrict_to_fold(multiclass_splits, args.fold)

    record_ids = resolve_record_ids(args.data_dir, args.records)

    multiclass_fold_results = [
        evaluate_multiclass_fold(args.train_outputs_dir, X_multi, y_multi, groups_multi, split, encoder, num_classes, window_size)
        for split in multiclass_splits["folds"]
    ]
    binary_fold_results = [
        evaluate_binary_fold(args.train_outputs_dir, X_bin, y_bin, groups_bin, split, window_size)
        for split in binary_splits["folds"]
    ]

    dataset_composition = section_dataset_composition(binary_splits, multiclass_splits, X_multi, y_multi, groups_multi, encoder)
    per_fold_table = section_per_fold_table(args.train_outputs_dir, multiclass_fold_results)
    aggregate_multiclass = section_aggregate_multiclass(multiclass_fold_results, encoder, num_classes)
    binary_clinical = section_binary_clinical_metrics(binary_fold_results)
    baselines = section_baselines(X_bin, y_bin, groups_bin, binary_splits, X_multi, y_multi, groups_multi, multiclass_splits, encoder, num_classes, run_params["random_seed"])
    pan_tompkins_rows, pan_tompkins_aggregate = evaluate_pan_tompkins(args.data_dir, record_ids, args.tolerance_samples)
    cascade_section = section_cascade_end_to_end(
        args.data_dir, args.train_outputs_dir, binary_splits, multiclass_splits,
        encoder, window_size, args.tolerance_samples,
    )
    gan_vs_nogan = section_gan_vs_nogan(args.train_outputs_dir, args.train_outputs_nogan_dir)

    binary_model_path = os.path.join(args.train_outputs_dir, "best_overall_binary_model.keras")
    multiclass_model_path = os.path.join(args.train_outputs_dir, "best_overall_multiclass_model.keras")
    binary_model = tf.keras.models.load_model(binary_model_path, custom_objects=CUSTOM_OBJECTS)
    multiclass_model = tf.keras.models.load_model(multiclass_model_path, custom_objects=CUSTOM_OBJECTS)
    binary_scaler = joblib.load(os.path.join(args.train_outputs_dir, "best_overall_binary_scaler.pkl"))
    cpu_benchmarks = section_cpu_benchmarks(
        args.data_dir, record_ids, binary_model, binary_scaler, multiclass_model, window_size,
        binary_model_path, multiclass_model_path,
    )

    cm_reordered, cm_labels = reorder_confusion_matrix(aggregate_multiclass["confusion_matrix"], list(encoder.classes_), CLINICAL_CLASS_ORDER)
    plot_confusion_matrix_figure(cm_reordered, cm_labels, os.path.join(img_dir, "confusion_matrix.png"), "Multiclass confusion matrix (row-normalized)")

    metrics_df = pd.read_csv(os.path.join(args.train_outputs_dir, "metrics.csv"))
    plot_per_fold_metrics_figure(metrics_df, os.path.join(img_dir, "per_fold_metrics.png"))

    if record_ids:
        plot_pan_tompkins_detection_figure(args.data_dir, record_ids[0], os.path.join(img_dir, "pan_tompkins_detection.png"))

    plot_wgan_real_vs_synthetic_figure(
        args.train_outputs_dir, X_multi, y_multi, groups_multi, multiclass_splits, encoder, window_size,
        os.path.join(img_dir, "wgan_real_vs_synthetic.png"),
    )

    sections = {
        "cascade": cascade_section,
        "aggregate_multiclass": aggregate_multiclass,
        "per_fold_table": per_fold_table,
        "binary_clinical": binary_clinical,
        "baselines": baselines,
        "gan_vs_nogan": gan_vs_nogan,
        "dataset_composition": dataset_composition,
        "cpu_benchmarks": cpu_benchmarks,
        "pan_tompkins": (pan_tompkins_rows, pan_tompkins_aggregate),
    }

    report_md = build_markdown_report(sections)
    with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as handle:
        handle.write(report_md)

    challenger_metrics = {
        "cascade_macro_f1": cascade_section["report"]["macro avg"]["f1-score"],
        "cv_mean_val_auc": mlflow.get_run(args.train_job_name).data.metrics.get("cv_mean_val_auc"),
        "binary_sensitivity": binary_clinical["aggregate"]["sensitivity"],
        "binary_specificity": binary_clinical["aggregate"]["specificity"],
    }

    metrics_json = _json_safe({
        "challenger_metrics": challenger_metrics,
        "cascade": cascade_section,
        "aggregate_multiclass": aggregate_multiclass,
        "per_fold_table": per_fold_table,
        "binary_clinical": binary_clinical,
        "baselines": baselines,
        "gan_vs_nogan": gan_vs_nogan,
        "dataset_composition": dataset_composition,
        "cpu_benchmarks": cpu_benchmarks,
        "pan_tompkins": {"per_record": pan_tompkins_rows, "aggregate": pan_tompkins_aggregate},
    })
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2)

    promotion_decision = build_promotion_decision(challenger_metrics, args.champion_model_name, args.metric_name)
    with open(os.path.join(output_dir, "promotion_decision.json"), "w", encoding="utf-8") as handle:
        json.dump(promotion_decision, handle, indent=2)

    try:
        for key, value in challenger_metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, value)
        mlflow.log_metric("cpu_batching_speedup_x", cpu_benchmarks["batching_speedup_x"])
        mlflow.log_metric("pan_tompkins_aggregate_sensitivity", pan_tompkins_aggregate["sensitivity"])
        mlflow.log_metric("pan_tompkins_aggregate_ppv", pan_tompkins_aggregate["ppv"])
        mlflow.log_artifact(os.path.join(output_dir, "evaluation_report.md"))
        mlflow.log_artifact(os.path.join(output_dir, "metrics.json"))
        mlflow.log_artifact(os.path.join(output_dir, "promotion_decision.json"))
        for fig_path in glob.glob(os.path.join(img_dir, "*.png")):
            mlflow.log_artifact(fig_path)
    except Exception as exc:
        print(f"Warning: mlflow logging failed ({exc}) - report/metrics/figures are still saved under {output_dir}/")

    print(f"Promotion decision: {promotion_decision['reason']}")


if __name__ == "__main__":
    main()
