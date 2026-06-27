import csv
from preprocessing import get_record_ids, data_loader
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from model import build_ecg_model
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import shutil
import pandas as pd


# ── Wykresy ───────────────────────────────────────────────────────────────────
def plot_loss(history, fold_number):
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss,     'bo-', label='Trening (Loss)')
    plt.plot(epochs, val_loss, 'ro-', label='Walidacja (Val Loss)')
    plt.title('Krzywa uczenia: Loss')
    plt.xlabel('Epoki')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/loss_curve_fold_{fold_number}.png')
    plt.close()  



def make_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Parameters:
        signal (np.ndarray): Input signal (1D array).
        window_size (int): Size of the sliding window.
        stride (int): Stride for sliding windows.
    Returns:
        np.ndarray: (n_windows, window_size, 1) 
    """
    sig = np.array(signal).squeeze()
    windows = sliding_window_view(sig, window_size)[::stride]
    return windows[..., np.newaxis].astype(np.float32)



def make_window_labels(labels: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Parameters:
        labels (np.ndarray): Input labels (0/1).
        window_size (int): Size of the sliding window.
        stride (int): Stride for sliding windows.
    Returns:
        np.ndarray: (n_windows,) — majority label for each window (0/1).
    """
    label_arr     = np.array(labels).squeeze().astype(int)
    label_windows = sliding_window_view(label_arr, window_size)[::stride]
    return np.apply_along_axis(
        lambda row: np.bincount(row).argmax(),
        axis=1,
        arr=label_windows
    )



base_dir   = Path(__file__).resolve().parent.parent
dir_path   = base_dir / 'data'
dir_path_str = str(dir_path)

record_ids               = get_record_ids(dir_path_str)
SELECTED_NUMBER_OF_SAMPLES = 20

MASTER_DATA, GROUPS = data_loader(dir_path_str, selected_number_of_samples=SELECTED_NUMBER_OF_SAMPLES)

SIGNAL = MASTER_DATA[['signal']]
LABELS = MASTER_DATA['label']
GROUPS = np.array(GROUPS)

FOLD_SPLITS = 5
WINDOW      = 4096
STRIDE      = 512

print(f"Rozmiar sygnału: {len(SIGNAL)}")
print(f"Rozmiar etykiet: {len(LABELS)}")
print(f"Rozmiar grup:    {len(GROUPS)}")
print(f"Rozkład klas:    {np.bincount(LABELS)}")

os.makedirs('outputs', exist_ok=True)


with open('outputs/metrics.csv', 'w', newline='') as f:
    csv.writer(f).writerow(['Fold', 'Best Epoch', 'Train Acc', 'Val Acc', 'Train Loss', 'Val Loss'])



fold_number = 0
for train_idx, test_idx in GroupKFold(n_splits=FOLD_SPLITS).split(SIGNAL, LABELS, GROUPS):
    print(f"\n{'='*50}\nFold {fold_number}\n{'='*50}")

    X_train_raw = SIGNAL.iloc[train_idx]
    X_test_raw  = SIGNAL.iloc[test_idx]

   
    y_train_raw = LABELS.iloc[train_idx].values
    y_test_raw  = LABELS.iloc[test_idx].values

    print(f"Trening - Wierszy: {len(X_train_raw)}, Grupy: {np.unique(GROUPS[train_idx])}")
    print(f"Test    - Wierszy: {len(X_test_raw)},  Grupy: {np.unique(GROUPS[test_idx])}")

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)
    joblib.dump(scaler, f'outputs/scaler_fold_{fold_number}.pkl')

    
    X_train_w = make_windows(X_train_scaled, WINDOW, STRIDE)
    y_train_w = make_window_labels(y_train_raw, WINDOW, STRIDE)

    X_test_w  = make_windows(X_test_scaled, WINDOW, STRIDE)
    y_test_w  = make_window_labels(y_test_raw,  WINDOW, STRIDE)

    print(f"Okna treningowe: {X_train_w.shape}, etykiety: {y_train_w.shape}")
    print(f"Okna testowe:    {X_test_w.shape},  etykiety: {y_test_w.shape}")
    print(f"Train - Klasy: {np.bincount(y_train_w)}")
    print(f"Val   - Klasy: {np.bincount(y_test_w)}")

    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_w)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    
    model = build_ecg_model(input_shape=(WINDOW, 1), n_classes=1)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        
        ModelCheckpoint(
            filepath=f'outputs/best_model_fold_{fold_number}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
    ]

    
    history = model.fit(
        X_train_w, y_train_w,
        epochs=30,
        validation_data=(X_test_w, y_test_w),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    
    best_epoch       = int(np.argmin(history.history['val_loss']))
    best_val_loss    = history.history['val_loss'][best_epoch]
    best_val_acc     = history.history['val_accuracy'][best_epoch]
    best_train_acc   = history.history['accuracy'][best_epoch]
    best_train_loss  = history.history['loss'][best_epoch]

    print(
        f"Fold {fold_number} | Najlepsza epoka: {best_epoch + 1} | "
        f"Train Acc={best_train_acc:.4f}, Loss={best_train_loss:.4f} | "
        f"Val Acc={best_val_acc:.4f}, Val Loss={best_val_loss:.4f}"
    )

    with open('outputs/metrics.csv', 'a', newline='') as f:
        csv.writer(f).writerow([
            fold_number, best_epoch + 1,
            best_train_acc, best_val_acc,
            best_train_loss, best_val_loss
        ])

    plot_loss(history, fold_number)
    fold_number += 1


metrics_df = pd.read_csv('outputs/metrics.csv')
best_fold  = int(metrics_df.loc[metrics_df['Val Loss'].idxmin(), 'Fold'])
best_val   = metrics_df['Val Loss'].min()

shutil.copy(
    f'outputs/best_model_fold_{best_fold}.keras',
    'outputs/best_overall_model.keras'
)
print(f"\n{'='*50}")
print(f"Najlepszy fold: {best_fold} | Val Loss: {best_val:.4f}")
print(f"Zapisano: outputs/best_overall_model.keras")