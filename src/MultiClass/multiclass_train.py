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


from multiclass_preprocessing import get_record_ids, get_multiclass_data
from multiclass_model import build_ecg_multiclass_model 

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import mlflow.keras


parser = argparse.ArgumentParser(description="ECG Multiclass Training Pipeline")
parser.add_argument("--data-dir", type=str, required=True, help="Path to raw ECG data")
parser.add_argument("--selected-samples", type=int, default=20, help="Number of ECG records to load")
parser.add_argument("--window", type=int, default=400, help="Fixed window size around the R-peak") 
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

args = parser.parse_args()


mlflow.log_params({
    "window_size": args.window,
    "epochs": args.epochs,
    "selected_samples": args.selected_samples,
    "random_seed": args.random_seed,
    "model_type": "Multiclass_1D_CNN"
})

class MLflowFoldCallback(tf.keras.callbacks.Callback):
    def __init__(self, fold_num):
        super().__init__()
        self.fold_num = fold_num

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                mlflow.log_metric(f"fold_{self.fold_num}_{key}", value, step=epoch)

def plot_loss(history, fold_number):
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss,     'bo-', label='Training (Loss)')
    plt.plot(epochs, val_loss, 'ro-', label='Validation (Val Loss)')
    plt.title('Learning Curve: Categorical Crossentropy Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/loss_curve_fold_{fold_number}.png')
    plt.close()  


def load_all_multiclass_data(data_dir: str, num_records: int):
    """
    Loops through the specified number of records using get_multiclass_data
    and generates the corresponding groups array for cross-validation.
    """
    X_all = []
    y_all = []
    groups_all = []
    
    record_ids = get_record_ids(data_dir)
    records_to_process = min(num_records, len(record_ids))
    
    print(f"\nExtracting anomaly windows from {records_to_process} records...")
    
    for i in range(records_to_process):
        X_record, y_record = get_multiclass_data(data_dir, sample_select=i)
        record_id = record_ids[i]
        
        if len(X_record) > 0:
            X_all.append(X_record)
            y_all.append(y_record)
            groups_all.extend([record_id] * len(X_record))
            
    X_master = np.vstack(X_all) if X_all else np.array([])
    y_master = np.concatenate(y_all) if y_all else np.array([])
    groups_master = np.array(groups_all)
    
    return X_master, y_master, groups_master

X_DATA, Y_LABELS, GROUPS = load_all_multiclass_data(args.data_dir, args.selected_samples)
os.makedirs('outputs', exist_ok=True)

encoder = LabelEncoder()
Y_ENCODED = encoder.fit_transform(Y_LABELS)
NUM_CLASSES = len(encoder.classes_)

joblib.dump(encoder, 'outputs/label_encoder.pkl')

print(f"\nTotal extracted anomalous windows: {len(X_DATA)}")
print(f"Detected {NUM_CLASSES} distinct anomaly classes: {encoder.classes_}")
print(f"Class distribution: {np.bincount(Y_ENCODED)}")

with open('outputs/metrics.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['Fold', 'Best Epoch', 'Train Acc', 'Val Acc', 'Train Loss', 'Val Loss', 'Val AUC'])

FOLD_SPLITS = 5
fold_number = 0

sgkf = StratifiedGroupKFold(n_splits=FOLD_SPLITS, shuffle=True, random_state=args.random_seed)

for train_idx, test_idx in sgkf.split(X_DATA, Y_ENCODED, GROUPS):
    print(f"\n{'='*50}\nFold {fold_number}\n{'='*50}")

    X_train_raw = X_DATA[train_idx]
    X_test_raw  = X_DATA[test_idx]
    
    y_train_raw = Y_ENCODED[train_idx]
    y_test_raw  = Y_ENCODED[test_idx]

    print(f"Training - Rows: {len(X_train_raw)}, Unique Patients: {len(np.unique(GROUPS[train_idx]))}")
    print(f"Test     - Rows: {len(X_test_raw)},  Unique Patients: {len(np.unique(GROUPS[test_idx]))}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    joblib.dump(scaler, f'outputs/scaler_fold_{fold_number}.pkl')

    X_train_w = X_train_scaled.reshape(-1, args.window, 1)
    X_test_w  = X_test_scaled.reshape(-1, args.window, 1)

    y_train_cat = to_categorical(y_train_raw, num_classes=NUM_CLASSES)
    y_test_cat  = to_categorical(y_test_raw, num_classes=NUM_CLASSES)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Computed Multiclass Weights: {class_weight_dict}")

    model = build_ecg_multiclass_model(input_shape=(args.window, 1), n_classes=NUM_CLASSES)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=f'outputs/best_model_fold_{fold_number}.keras', monitor='val_loss', save_best_only=True, verbose=1),
        MLflowFoldCallback(fold_num=fold_number)
    ]
    
    history = model.fit(
        X_train_w, y_train_cat,
        epochs=args.epochs,
        validation_data=(X_test_w, y_test_cat),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    best_epoch      = int(np.argmin(history.history['val_loss']))
    best_val_loss   = history.history['val_loss'][best_epoch]
    best_val_acc    = history.history['val_accuracy'][best_epoch]
    best_train_acc  = history.history['accuracy'][best_epoch]
    best_train_loss = history.history['loss'][best_epoch]

    
    y_pred_probs = model.predict(X_test_w)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    y_pred_probs = model.predict(X_test_w)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    try:
        present_classes = np.any(y_test_cat > 0, axis=0)
        best_val_auc = roc_auc_score(
            y_test_cat[:, present_classes],
            y_pred_probs[:, present_classes],
            multi_class='ovr',
            average='weighted'
        )
    except Exception as e:
        print(f"Warning: Failed to calculate weighted AUC for fold {fold_number}: {e}")
        best_val_auc = 0.5  
    
    cm = confusion_matrix(y_test_raw, y_pred, labels=range(NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    
    plt.title(f'Multiclass Confusion Matrix - Fold {fold_number}')
    plt.tight_layout()
    plt.savefig(f'outputs/confusion_matrix_fold_{fold_number}.png')
    plt.close(fig)

   
    mlflow.log_metric(f"best_val_loss_fold_{fold_number}", best_val_loss)
    mlflow.log_metric(f"best_val_acc_fold_{fold_number}", best_val_acc)
    mlflow.log_metric(f"best_val_auc_fold_{fold_number}", best_val_auc)

    print(f"Fold {fold_number} | Best epoch: {best_epoch + 1} | "
          f"Train Acc={best_train_acc:.4f}, Loss={best_train_loss:.4f} | "
          f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}, Val AUC={best_val_auc:.4f}")

    with open('outputs/metrics.csv', 'a', newline='') as f:
        csv.writer(f).writerow([
            fold_number, best_epoch + 1,
            best_train_acc, best_val_acc,
            best_train_loss, best_val_loss,
            best_val_auc
        ])

    plot_loss(history, fold_number)
    fold_number += 1


metrics_df = pd.read_csv('outputs/metrics.csv')
best_fold  = int(metrics_df.loc[metrics_df['Val Loss'].idxmin(), 'Fold'])
best_val   = metrics_df['Val Loss'].min()

mean_val_loss = metrics_df['Val Loss'].mean()
mean_val_acc = metrics_df['Val Acc'].mean()
mean_val_auc = metrics_df['Val AUC'].mean()

mlflow.log_metric("cv_mean_val_loss", mean_val_loss)
mlflow.log_metric("cv_mean_val_acc", mean_val_acc)
mlflow.log_metric("cv_mean_val_auc", mean_val_auc)
mlflow.log_metric("best_fold_number", best_fold)

shutil.copy(f'outputs/best_model_fold_{best_fold}.keras', 'outputs/best_overall_multiclass_model.keras')
shutil.copy(f'outputs/scaler_fold_{best_fold}.pkl', 'outputs/best_overall_scaler.pkl')

os.makedirs("./outputs", exist_ok=True)
model.save("./outputs/ecg_multiclass_model.keras")
print("Model saved successfully to ./outputs/ecg_multiclass_model.keras")

print(f"\n{'='*50}")
print(f"MULTICLASS TRAINING COMPLETE")
print(f"Mean CV Accuracy: {mean_val_acc:.4f} | Mean CV AUC (OVR): {mean_val_auc:.4f}")
print(f"Best fold: {best_fold} | Val Loss: {best_val:.4f}")
print(f"Saved best model: outputs/best_overall_multiclass_model.keras")
print(f"Saved Label Encoder: outputs/label_encoder.pkl")