import os
import csv
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from multistage_preprocessing import get_record_ids, get_data, get_global_window_size

from multistage_model import build_inception_conformer
from data_augment import augment_ecg
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import mlflow.keras


class MLflowFoldCallback(tf.keras.callbacks.Callback):
    def __init__(self, fold_num, stage):
        super().__init__()
        self.fold_num = fold_num
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                mlflow.log_metric(f"fold_{self.fold_num}_{self.stage}_{key}", value, step=epoch)


def build_parser():
    parser = argparse.ArgumentParser(description="ECG Multistage Training Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw ECG data")
    parser.add_argument("--selected-samples", type=int, default=20, help="Number of ECG records to load")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start_learning_rate", type=float, default=1.5e-3, help="Initial learning rate for the optimizer")
    parser.add_argument("--Folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["binary", "multiclass", "both"],
        help="Which training pipeline to run",
    )
    return parser


def plot_loss(history, fold_number, stage):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, "bo-", label="Training (Loss)")
    plt.plot(epochs, val_loss, "ro-", label="Validation (Val Loss)")
    plt.title(f"Learning Curve: Loss - Stage {stage}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"outputs/loss_curve_fold_{fold_number}_stage_{stage}.png")
    plt.close()

def samples_plot(X_data, y_labels, encoder, stage, random_state):
    """
    Plots a random sample from each class in the dataset.
    Parameters:
        X_data (np.array): 2D array of ECG windows.
        y_labels (np.array): 1D array of corresponding labels.
        encoder (LabelEncoder): Encoder to transform labels back to original class names.
        stage (str): The stage of the classification task ('binary' or 'multiclass').
        random_state (int): Seed for reproducibility of random sampling.
    """
    rng = np.random.default_rng(random_state)

    if stage == "binary":
        class_indices = [0, 1]
        class_names = ["N", "anomaly"]
    else:
        class_indices = list(range(len(encoder.classes_)))
        class_names = list(encoder.classes_)

    selected_indices = []
    selected_titles = []

    for class_index, class_name in zip(class_indices, class_names):
        candidate_indices = np.where(y_labels == class_index)[0]
        if len(candidate_indices) == 0:
            continue
        chosen_index = int(rng.choice(candidate_indices))
        selected_indices.append(chosen_index)
        selected_titles.append(class_name)

    if not selected_indices:
        print(f"No samples available to plot for stage '{stage}'.")
        return

    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(12, 3 * len(selected_indices)), sharex=True)
    if len(selected_indices) == 1:
        axes = [axes]

    for axis, sample_index, title in zip(axes, selected_indices, selected_titles):
        axis.plot(X_data[sample_index])
        axis.set_title(f"{stage.capitalize()} sample: {title}")
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/sample_preview_{stage}.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved sample preview to {output_path}")


def load_data(data_dir: str, num_records: int, random_seed: int, stage: str):
    """
    Loads ECG signal windows and their corresponding labels from the specified directory.
    Applies dynamic downsampling based on interquartile range (IQR) to balance the dataset.
    """
    X_all, y_all, groups_all = [], [], []
    record_ids = get_record_ids(data_dir)
    records_to_process = min(num_records, len(record_ids))
    window_size = get_global_window_size(data_dir, record_ids)
    print(f"Global window size determined: {window_size}")
    for i in range(records_to_process):
        X_record, y_record = get_data(data_dir, sample_select=i, stage=stage, window_size=window_size)
        if len(X_record) > 0:
            X_all.append(X_record)
            y_all.append(y_record)
            groups_all.extend([record_ids[i]] * len(X_record))

    X_master = np.vstack(X_all) if X_all else np.array([])
    y_master = np.concatenate(y_all) if y_all else np.array([])
    groups_master = np.array(groups_all)

    if len(y_master) == 0:
        return X_master, y_master, groups_master

    rng = np.random.default_rng(random_seed)
    valid_indices = []

    for cls in np.unique(y_master):
        cls_mask = (y_master == cls)
        patients_with_cls = groups_master[cls_mask]

        unique_patients, counts = np.unique(patients_with_cls, return_counts=True)

        q1 = np.percentile(counts, 25)
        q3 = np.percentile(counts, 75)
        iqr = q3 - q1

        dynamic_limit = int(np.ceil(q3 + 1.5 * iqr))
        dynamic_limit = max(dynamic_limit, 30)

        for patient in unique_patients:
            patient_cls_idx = np.where((y_master == cls) & (groups_master == patient))[0]

            if len(patient_cls_idx) > dynamic_limit:
                patient_cls_idx = rng.choice(patient_cls_idx, size=dynamic_limit, replace=False)

            valid_indices.extend(patient_cls_idx)

    valid_indices = np.sort(valid_indices)
    return X_master[valid_indices], y_master[valid_indices], groups_master[valid_indices], window_size


def train_binary_stage(args):
    print("STAGE 1: BINARY CLASSIFICATION (Normal vs. Abnormal)")

    X_data, y_labels, groups, window_size = load_data(args.data_dir, args.selected_samples, args.random_seed, stage="binary")
    X_data = X_data.astype(np.float32)

    samples_plot(X_data, y_labels, encoder=None, stage="binary", random_state=args.random_seed)

    os.makedirs("outputs", exist_ok=True)

    y_encoded = y_labels.astype(np.int32)
    num_classes = 2

    print(f"\nTotal extracted windows: {len(X_data)}")
    print(f"Detected {num_classes} distinct classes: [0, 1]")
    print(f"Class distribution: {np.bincount(y_encoded)}")

    with open("outputs/metrics_binary.csv", "w", newline="") as f:
        csv.writer(f).writerow(["Fold", "Best Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss", "Val AUC", "F1 Score"])

    sgkf = StratifiedGroupKFold(n_splits=args.Folds, shuffle=True, random_state=args.random_seed)
    fold_number = 0

    for train_idx, test_idx in sgkf.split(X_data, y_encoded, groups):
        print(f"\n{'=' * 50}\nFold {fold_number}\n{'=' * 50}")

        X_train_raw = X_data[train_idx]
        X_test_raw = X_data[test_idx]
        y_train_raw = y_encoded[train_idx]
        y_test_raw = y_encoded[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        joblib.dump(scaler, f"outputs/scaler_fold_{fold_number}_binary.pkl")

        X_train_w = X_train_scaled.reshape(-1, window_size, 1)
        X_test_w = X_test_scaled.reshape(-1, window_size, 1)

        y_train_bin = y_train_raw.astype(np.float32).reshape(-1, 1)
        y_test_bin = y_test_raw.astype(np.float32).reshape(-1, 1)

        model = build_inception_conformer(window_size=window_size, n_classes=num_classes, stage="binary")
        steps_per_epoch = len(X_train_w) // args.batch_size
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(0.1 * total_steps)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.start_learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=0.01,
            warmup_target=args.start_learning_rate,
            warmup_steps=warmup_steps,
        )

        focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=0.25,
            gamma=1,
            # label_smoothing=0.05
        )

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0, weight_decay=1e-2),
            loss=focal_loss,
            metrics=["accuracy", tf.keras.metrics.F1Score(average="macro", name="macro_f1")],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=f"outputs/best_model_fold_{fold_number}_binary.keras", monitor="val_loss", save_best_only=True, verbose=1),
            MLflowFoldCallback(fold_num=fold_number, stage="binary"),
        ]

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train_w, y_train_bin))
            .shuffle(buffer_size=1024)
            .map(augment_ecg, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_test_w, y_test_bin))
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = int(np.argmin(history.history["val_loss"]))
        best_val_loss = history.history["val_loss"][best_epoch]
        best_val_acc = history.history["val_accuracy"][best_epoch]
        best_train_acc = history.history["accuracy"][best_epoch]
        best_train_loss = history.history["loss"][best_epoch]
        best_f1_score = history.history["macro_f1"][best_epoch]

        y_pred_probs = model.predict(X_test_w, verbose=0).ravel()
        y_pred = (y_pred_probs >= 0.5).astype(np.int32)

        try:
            best_val_auc = roc_auc_score(
                y_test_raw,
                y_pred_probs,
            )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: Failed to calculate weighted AUC for fold {fold_number}: {exc}")
            best_val_auc = 0.5

        cm = confusion_matrix(y_test_raw, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["N", "anomaly"])
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
        plt.title(f"BINARY Confusion Matrix - Fold {fold_number}")
        plt.tight_layout()
        plt.savefig(f"outputs/confusion_matrix_fold_{fold_number}_binary.png")
        plt.close(fig)

        report_text = classification_report(
            y_test_raw,
            y_pred,
            labels=[0, 1],
            target_names=["N", "anomaly"],
            zero_division=0,
        )
        with open(f"outputs/classification_report_fold_{fold_number}_binary.txt", "w", encoding="utf-8") as handle:
            handle.write(f"Detailed Binary Classification Report - Fold {fold_number} ===\n\n")
            handle.write(report_text)

        mlflow.log_metric(f"best_binary_val_loss_fold_{fold_number}", best_val_loss)
        mlflow.log_metric(f"best_binary_val_acc_fold_{fold_number}", best_val_acc)
        mlflow.log_metric(f"best_binary_val_auc_fold_{fold_number}", best_val_auc)
        mlflow.log_metric(f"best_binary_f1_score_fold_{fold_number}", best_f1_score)

        print(
            f"Fold {fold_number} | Best epoch: {best_epoch + 1} | "
            f"Train Acc={best_train_acc:.4f}, Loss={best_train_loss:.4f} | "
            f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}, Val AUC={best_val_auc:.4f}, F1 Score={best_f1_score:.4f}"
        )

        with open("outputs/metrics_binary.csv", "a", newline="") as handle:
            csv.writer(handle).writerow([
                fold_number,
                best_epoch + 1,
                best_train_acc,
                best_val_acc,
                best_train_loss,
                best_val_loss,
                best_val_auc,
                best_f1_score,
            ])

        plot_loss(history, fold_number, stage="binary")
        fold_number += 1

    metrics_df = pd.read_csv("outputs/metrics_binary.csv")
    best_fold = int(metrics_df.loc[metrics_df["Val Loss"].idxmin(), "Fold"])
    best_val = metrics_df["Val Loss"].min()
    mean_val_loss = metrics_df["Val Loss"].mean()
    mean_val_acc = metrics_df["Val Acc"].mean()
    mean_val_auc = metrics_df["Val AUC"].mean()
    mean_f1_score = metrics_df["F1 Score"].mean()

    mlflow.log_metric("cv_binary_mean_val_loss", mean_val_loss)
    mlflow.log_metric("cv_binary_mean_val_acc", mean_val_acc)
    mlflow.log_metric("cv_binary_mean_val_auc", mean_val_auc)
    mlflow.log_metric("cv_binary_mean_f1_score", mean_f1_score)
    mlflow.log_metric("best_binary_fold_number", best_fold)

    shutil.copy(f"outputs/best_model_fold_{best_fold}_binary.keras", "outputs/best_overall_binary_model.keras")
    shutil.copy(f"outputs/scaler_fold_{best_fold}_binary.pkl", "outputs/best_overall_binary_scaler.pkl")

    model.save("outputs/ecg_binary_model.keras")
    print("Model saved successfully to outputs/ecg_binary_model.keras")

    print(f"\n{'=' * 50}")
    print("BINARY TRAINING COMPLETE")
    print(f"Mean CV Accuracy: {mean_val_acc:.4f} | Mean CV AUC (OVR): {mean_val_auc:.4f} | Mean CV F1 Score: {mean_f1_score:.4f}")
    print(f"Best fold: {best_fold} | Val Loss: {best_val:.4f}")
    print("Saved best binary model: outputs/best_overall_binary_model.keras")


def train_multiclass_stage(args):
    print("STAGE 2: MULTICLASS CLASSIFICATION")

    X_data, y_labels, groups, window_size = load_data(args.data_dir, args.selected_samples, args.random_seed, stage="multiclass")
    X_data = X_data.astype(np.float32)

    os.makedirs("outputs", exist_ok=True)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)
    num_classes = len(encoder.classes_)

    joblib.dump(encoder, "outputs/label_encoder.pkl")

    samples_plot(X_data, y_encoded, encoder=encoder, stage="multiclass", random_state=args.random_seed)

    print(f"\nTotal extracted windows: {len(X_data)}")
    print(f"Detected {num_classes} distinct classes: {encoder.classes_}")
    print(f"Class distribution: {np.bincount(y_encoded)}")

    with open("outputs/metrics.csv", "w", newline="") as f:
        csv.writer(f).writerow(["Fold", "Best Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss", "Val AUC", "F1 Score"])

    sgkf = StratifiedGroupKFold(n_splits=args.Folds, shuffle=True, random_state=args.random_seed)
    fold_number = 0

    for train_idx, test_idx in sgkf.split(X_data, y_encoded, groups):
        print(f"\n{'=' * 50}\nFold {fold_number}\n{'=' * 50}")

        X_train_raw = X_data[train_idx]
        X_test_raw = X_data[test_idx]
        y_train_raw = y_encoded[train_idx]
        y_test_raw = y_encoded[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        joblib.dump(scaler, f"outputs/scaler_fold_{fold_number}.pkl")

        X_train_w = X_train_scaled.reshape(-1, window_size, 1)
        X_test_w = X_test_scaled.reshape(-1, window_size, 1)

        y_train_cat = to_categorical(y_train_raw, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_raw, num_classes=num_classes)

        model = build_inception_conformer(window_size=window_size, n_classes=num_classes, stage="multiclass")
        steps_per_epoch = len(X_train_w) // args.batch_size
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(0.1 * total_steps)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.start_learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=0.01,
            warmup_target=args.start_learning_rate,
            warmup_steps=warmup_steps,
        )

        focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25,
            gamma=1,
            label_smoothing=0.05,
        )

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0, weight_decay=1e-2),
            loss=focal_loss,
            metrics=["accuracy", tf.keras.metrics.F1Score(average="macro", name="macro_f1")],
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, min_delta=1e-4, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=f"outputs/best_model_fold_{fold_number}.keras", monitor="val_loss", save_best_only=True, verbose=1),
            MLflowFoldCallback(fold_num=fold_number, stage="multiclass"),
        ]

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train_w, y_train_cat))
            .shuffle(buffer_size=1024)
            .map(augment_ecg, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_test_w, y_test_cat))
            .batch(args.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = int(np.argmin(history.history["val_loss"]))
        best_val_loss = history.history["val_loss"][best_epoch]
        best_val_acc = history.history["val_accuracy"][best_epoch]
        best_train_acc = history.history["accuracy"][best_epoch]
        best_train_loss = history.history["loss"][best_epoch]
        best_f1_score = history.history["macro_f1"][best_epoch]

        y_pred_probs = model.predict(X_test_w, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        try:
            present_classes = np.any(y_test_cat > 0, axis=0)
            best_val_auc = roc_auc_score(
                y_test_cat[:, present_classes],
                y_pred_probs[:, present_classes],
                average="weighted",
                multi_class="ovr",
            )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: Failed to calculate weighted AUC for fold {fold_number}: {exc}")
            best_val_auc = 0.5

        cm = confusion_matrix(y_test_raw, y_pred, labels=range(num_classes))
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
        plt.title(f"MULTICLASS Confusion Matrix - Fold {fold_number}")
        plt.tight_layout()
        plt.savefig(f"outputs/confusion_matrix_fold_{fold_number}.png")
        plt.close(fig)

        report_text = classification_report(
            y_test_raw,
            y_pred,
            labels=range(num_classes),
            target_names=list(encoder.classes_),
            zero_division=0,
        )
        with open(f"outputs/classification_report_fold_{fold_number}.txt", "w", encoding="utf-8") as handle:
            handle.write(f"Detailed Classification Report - Fold {fold_number} ===\n\n")
            handle.write(report_text)

        mlflow.log_metric(f"best_val_loss_fold_{fold_number}", best_val_loss)
        mlflow.log_metric(f"best_val_acc_fold_{fold_number}", best_val_acc)
        mlflow.log_metric(f"best_val_auc_fold_{fold_number}", best_val_auc)
        mlflow.log_metric(f"best_f1_score_fold_{fold_number}", best_f1_score)

        print(
            f"Fold {fold_number} | Best epoch: {best_epoch + 1} | "
            f"Train Acc={best_train_acc:.4f}, Loss={best_train_loss:.4f} | "
            f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}, Val AUC={best_val_auc:.4f}, F1 Score={best_f1_score:.4f}"
        )

        with open("outputs/metrics.csv", "a", newline="") as handle:
            csv.writer(handle).writerow([
                fold_number,
                best_epoch + 1,
                best_train_acc,
                best_val_acc,
                best_train_loss,
                best_val_loss,
                best_val_auc,
                best_f1_score,
            ])

        plot_loss(history, fold_number, stage="multiclass")
        fold_number += 1

    metrics_df = pd.read_csv("outputs/metrics.csv")
    best_fold = int(metrics_df.loc[metrics_df["Val Loss"].idxmin(), "Fold"])
    best_val = metrics_df["Val Loss"].min()
    mean_val_loss = metrics_df["Val Loss"].mean()
    mean_val_acc = metrics_df["Val Acc"].mean()
    mean_val_auc = metrics_df["Val AUC"].mean()
    mean_f1_score = metrics_df["F1 Score"].mean()

    mlflow.log_metric("cv_mean_val_loss", mean_val_loss)
    mlflow.log_metric("cv_mean_val_acc", mean_val_acc)
    mlflow.log_metric("cv_mean_val_auc", mean_val_auc)
    mlflow.log_metric("cv_mean_f1_score", mean_f1_score)
    mlflow.log_metric("best_fold_number", best_fold)

    shutil.copy(f"outputs/best_model_fold_{best_fold}.keras", "outputs/best_overall_multiclass_model.keras")
    shutil.copy(f"outputs/scaler_fold_{best_fold}.pkl", "outputs/best_overall_scaler.pkl")

    model.save("outputs/ecg_multiclass_model.keras")
    print("Model saved successfully to outputs/ecg_multiclass_model.keras")

    print(f"\n{'=' * 50}")
    print("MULTICLASS TRAINING COMPLETE")
    print(f"Mean CV Accuracy: {mean_val_acc:.4f} | Mean CV AUC (OVR): {mean_val_auc:.4f} | Mean CV F1 Score: {mean_f1_score:.4f}")
    print(f"Best fold: {best_fold} | Val Loss: {best_val:.4f}")
    print("Saved best model: outputs/best_overall_multiclass_model.keras")
    print("Saved Label Encoder: outputs/label_encoder.pkl")


def run_selected_task(args):
    if args.task in {"binary", "both"}:
        train_binary_stage(args)
    if args.task in {"multiclass", "both"}:
        train_multiclass_stage(args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    mlflow.log_params({
        "epochs": args.epochs,
        "selected_samples": args.selected_samples,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "Folds": args.Folds,
        "task": args.task,
        "model_type": "Multistage Inception-Conformer",
    })

    run_selected_task(args)


if __name__ == "__main__":
    main()
