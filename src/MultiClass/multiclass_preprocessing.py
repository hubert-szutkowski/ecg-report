import numpy as np
import wfdb
from pathlib import Path
import os

NORMAL_SYMBOLS = {'N', '/', 'f', '', ' ', '"'}

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


def extract_anomaly_windows(signals, features_samples, features, window_back=150, window_forward=250):
    """
    Slicing each anomaly into a window of samples around the peak. The window is defined by the number of samples before and after the peak.
    
    Parametry:
        signals (np.array): Complete ECG signal from which to extract windows.
        features_samples (list/np.array): Indices of the peaks in the signal corresponding to the features.
        features (list/np.array): Labels of the features corresponding to the peaks (e.g., 'V', 'A', 'N').
        window_back (int): Number of samples to extract before the peak.
        window_forward (int): Number of samples to extract after the peak.

    Returns:
        X (np.array): 2D array with the extracted windows [number_of_anomalies, window_width].
        y (np.array): 1D array with the corresponding anomaly labels (e.g., 'V', 'A').
    """
    X_windows = []
    y_labels = []
    n_samples = len(signals)

    for sample_pos, symbol in zip(features_samples, features):
        #Ignoring normal symbols, we only want to extract anomalies
        if symbol in NORMAL_SYMBOLS:
            continue
            
        #Calculating the window boundaries around the peak
        start_idx = sample_pos - window_back
        end_idx = sample_pos + window_forward
        
        #Protection against signal edges (beginning/end of the recording)
        if start_idx >= 0 and end_idx < n_samples:
            # Slicing the signal to get the window around the peak
            window = signals[start_idx:end_idx].flatten()
            
            X_windows.append(window)
            y_labels.append(symbol)
            
    return np.array(X_windows), np.array(y_labels)

def get_multiclass_data(dir_path: str, sample_select: int = 0):
    """
    Getting a specific record from the data directory and extracting anomaly windows.
    Parameters:
        dir_path (str): The path to the data directory.
        sample_select (int): The index of the record to select.
    Returns:
        X (np.array): 2D array with the extracted windows [number_of_anomalies, window_width].
        y (np.array): 1D array with the corresponding anomaly labels (e.g., 'V', 'A').
    """
    records_ids  = get_record_ids(dir_path)
    record_path  = str(Path(dir_path) / str(records_ids[sample_select]))

    ecg_annotations  = wfdb.rdann(record_path, 'atr')
    features         = np.array(ecg_annotations.__dict__['symbol'])
    features_samples = ecg_annotations.__dict__['sample']

    signals, _ = wfdb.rdsamp(record_path, channels=[0])
    
    X, y = extract_anomaly_windows(signals, features_samples, features)
    
    print(
        f"Record {records_ids[sample_select]:>6} | "
        f"Anomaly removed: {len(y)} | "
        f"Unique labels: {np.unique(y)}"
    )
    
    return X, y