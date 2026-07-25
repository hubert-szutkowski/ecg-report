import numpy as np
import wfdb
from pathlib import Path
import os

NORMAL = {'N', 'L', 'R', 'e', 'j'}
SVEB = {'A', 'a', 'J', 'S'}
VEB = {'V', 'E'}
FUSION = {'F'}
UNKNOWN = {'/', 'f', 'Q', '?', " "}

# Map raw symbols to AAMI classes
SYMBOL_TO_CLASS = {}
SYMBOL_TO_CLASS.update({s: 'N' for s in NORMAL})
SYMBOL_TO_CLASS.update({s: 'S' for s in SVEB})
SYMBOL_TO_CLASS.update({s: 'V' for s in VEB})
SYMBOL_TO_CLASS.update({s: 'F' for s in FUSION})
SYMBOL_TO_CLASS.update({s: 'Q' for s in UNKNOWN})


def get_record_ids(data_dir: str) -> list:
    """
    Getting the record IDs from the data directory.

    Parameters:
        data_dir (str): The path to the data directory.
    Returns:
        list: A sorted list of record IDs.
    """
    files = os.listdir(data_dir)
    record_ids = {f.split('.')[0] for f in files if f.split('.')[0].isdigit()}
    return sorted(list(record_ids))


def extract_AAMI_windows(signals, features_samples, features, window_back=150, window_forward=250):
    """
    Slicing each anomaly into a window of samples around the peak. The window is defined by the number of samples before and after the peak.
    The label in y is the single-letter AAMI class (S/V/F/Q), not the raw annotation symbol.

    Parameters:
        signals (np.array): Complete ECG signal from which to extract windows.
        features_samples (list/np.array): Indices of the peaks in the signal corresponding to the features.
        features (list/np.array): Labels of the features corresponding to the peaks (e.g., 'V', 'A', 'N').
        window_back (int): Number of samples to extract before the peak.
        window_forward (int): Number of samples to extract after the peak.

    Returns:
        X (np.array): 2D array with the extracted windows [number_of_anomalies, window_width].
        y (np.array): 1D array with the corresponding AAMI class labels ('S', 'V', 'F', 'Q').
    """
    X_windows = []
    y_labels = []
    n_samples = len(signals)

    for sample_pos, symbol in zip(features_samples, features):
        
        # Skip unmapped symbols
        aami_class = SYMBOL_TO_CLASS.get(symbol)
        if aami_class is None:
            continue

        # Compute window bounds
        start_idx = sample_pos - window_back
        end_idx = sample_pos + window_forward

        # Skip edge cases
        if start_idx >= 0 and end_idx < n_samples:
            # Slice the signal window
            window = signals[start_idx:end_idx].flatten()

            X_windows.append(window)
            y_labels.append(aami_class)

    return np.array(X_windows), np.array(y_labels)


def get_data(dir_path: str, sample_select: int = 0, stage: str = 'binary'):
    """
    Getting a specific record from the data directory and extracting AAMI windows.
    Parameters:
        dir_path (str): The path to the data directory.
        sample_select (int): The index of the record to select.
        stage (str): The stage of the classification task ('binary' or 'multiclass').
    Returns:
        X (np.array): 2D array with the extracted windows [number_of_anomalies, window_width].
        y (np.array):
            - for 'multiclass': 1D array with arrhythmia labels ('S', 'V', 'F', 'Q'), with 'N' removed
            - for 'binary': 1D array with integer labels (0 for N, 1 for anomaly)
    """
    records_ids  = get_record_ids(dir_path)
    record_path  = str(Path(dir_path) / str(records_ids[sample_select]))

    ecg_annotations  = wfdb.rdann(record_path, 'atr')
    features         = np.array(ecg_annotations.__dict__['symbol'])
    features_samples = ecg_annotations.__dict__['sample']

    signals, _ = wfdb.rdsamp(record_path, channels=[0])

    X, y = extract_AAMI_windows(signals, features_samples, features)

    if stage == 'binary':
        # Binary target
        y = (y != 'N').astype(np.int32)
    elif stage == 'multiclass':
        # Drop normal beats for multiclass
        arrhythmia_mask = (y != 'N')
        X = X[arrhythmia_mask]
        y = y[arrhythmia_mask]
    elif stage != 'multiclass':
        raise ValueError(f"Unsupported stage '{stage}'. Use 'binary' or 'multiclass'.")

    print(
        f"Record {records_ids[sample_select]:>6} | "
        f"Extracted windows: {len(y)} | "
        f"Unique labels: {np.unique(y)}"
    )

    return X, y