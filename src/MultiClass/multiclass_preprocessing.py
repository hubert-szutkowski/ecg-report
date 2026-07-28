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


def extract_anomaly_windows(signals, features_samples, features, window_size=216, normal_symbols=NORMAL, symbol_to_class=SYMBOL_TO_CLASS):
    """
    Slicing each anomaly into a window of samples around the peak. The window is defined by the number of samples before and after the peak.
    The label in y is the AAMI class (S/V/F/Q), not the raw symbol.

    Parameters:
        signals (np.array): Complete ECG signal from which to extract windows.
        features_samples (list/np.array): Indices of the peaks in the signal corresponding to the features.
        features (list/np.array): Labels of the features corresponding to the peaks (e.g., 'V', 'A', 'N').
        window_size (int): Total number of samples in the extracted window.
        normal_symbols (set): Set of symbols representing normal beats.
        symbol_to_class (dict): Mapping from raw symbols to AAMI classes.

    Returns:
        X (np.array): 2D array with the extracted windows [number_of_anomalies, window_width].
        y (np.array): 1D array with the corresponding AAMI class labels ('S', 'V', 'F', 'Q').
    """
    X_windows = []
    y_labels = []
    n_samples = len(signals)
    
    
    half_window = window_size // 2

    for sample_pos, symbol in zip(features_samples, features):
        
        if symbol in normal_symbols:
            continue

        aami_class = symbol_to_class.get(symbol)
        if aami_class is None:
            continue

        
        start_idx = sample_pos - half_window
        end_idx = start_idx + window_size

        
        if start_idx >= 0 and end_idx <= n_samples:
            window = signals[start_idx:end_idx].flatten()
            
            X_windows.append(window)
            y_labels.append(aami_class)

    return np.array(X_windows), np.array(y_labels)


def get_multiclass_data(dir_path: str, sample_select: int = 0, window_size: int = 216) -> tuple:
    """
    Getting a specific record from the data directory and extracting anomaly windows.
    Parameters:
        dir_path (str): The path to the data directory.
        sample_select (int): The index of the record to select.
    Returns:
        X (np.array): 2D array with the extracted windows [number_of_anomalies, window_width].
        y (np.array): 1D array with the corresponding AAMI class labels ('S', 'V', 'F', 'Q').
    """
    records_ids  = get_record_ids(dir_path)
    record_path  = str(Path(dir_path) / str(records_ids[sample_select]))

    ecg_annotations  = wfdb.rdann(record_path, 'atr')
    features         = np.array(ecg_annotations.__dict__['symbol'])
    features_samples = ecg_annotations.__dict__['sample']

    signals, _ = wfdb.rdsamp(record_path, channels=[0])

    X, y = extract_anomaly_windows(signals, features_samples, features, window_size)

    print(
        f"Record {records_ids[sample_select]:>6} | "
        f"Anomaly removed: {len(y)} | "
        f"Unique labels: {np.unique(y)}"
    )

    return X, y


def get_global_window_size(dir_path: str, record_ids: list, scale_factor: float = 0.8) -> int:
    all_rr_distances = []
    
    for record in record_ids:
        record_path = str(Path(dir_path) / str(record))
        ecg_annotations = wfdb.rdann(record_path, 'atr')
        
        features_samples = ecg_annotations.sample
        rr_distances = np.diff(features_samples)
        all_rr_distances.extend(rr_distances)
        
    all_rr_distances = np.array(all_rr_distances)
    valid_rr_distances = all_rr_distances[all_rr_distances > 0]
    
    
    raw_median = np.median(valid_rr_distances)
    global_seq_len = int(np.floor(raw_median * scale_factor))
    
    
    if global_seq_len % 2 != 0:
        global_seq_len -= 1
        
    return global_seq_len
