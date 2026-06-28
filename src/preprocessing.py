import os
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path


NORMAL_SYMBOL = 'N'


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


def get_record(dir_path: str, sample_select: int = 0) -> pd.DataFrame:
    """
    Getting a specific record from the data directory.
    Labels are binary: N=0, others=1 — consistent for EACH patient.


    Parameters:
        dir_path (str): The path to the data directory.
        sample_select (int): The index of the record to select.
    Returns:
        pd.DataFrame: DataFrame with columns 'signal' (float) and 'label' (0/1).
    """
    records_ids  = get_record_ids(dir_path)
    record_path  = str(Path(dir_path) / str(records_ids[sample_select]))

    ecg_annotations  = wfdb.rdann(record_path, 'atr')
    features         = np.array(ecg_annotations.__dict__['symbol'])
    features_samples = ecg_annotations.__dict__['sample']

    signals, _ = wfdb.rdsamp(record_path, channels=[0])
    n_samples  = len(signals)

    
    labels = np.zeros(n_samples, dtype=int)

    for i, (sample_pos, symbol) in enumerate(zip(features_samples, features)):
        label_value = 0 if symbol == NORMAL_SYMBOL else 1

        start = sample_pos
        end   = features_samples[i + 1] if i + 1 < len(features_samples) else n_samples
        labels[start:end] = label_value

    
    n_normal  = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()
    unique_symbols = np.unique(features)
    print(
        f"Record {records_ids[sample_select]:>6} | "
        f"Symbols: {sorted(unique_symbols)} | "
        f"N={n_normal} ({n_normal/n_samples*100:.1f}%) | "
        f"anomaly={n_anomaly} ({n_anomaly/n_samples*100:.1f}%)"
    )

    return pd.DataFrame({'signal': signals.flatten(), 'label': labels})


def data_loader(dir_path: str, selected_number_of_samples: int = 20) -> tuple:
    """
    Loading multiple records and combining them into a single DataFrame.

    Parameters:
        dir_path (str): The path to the data directory.
        selected_number_of_samples (int): The number of records to load.
    Returns:
        tuple: (master_data DataFrame, groups list)
    """
    records_list = []
    for i in range(selected_number_of_samples):
        record_df = get_record(dir_path, sample_select=i)
        records_list.append(record_df)

    all_data = []
    groups   = []
    for i, record in enumerate(records_list):
        all_data.append(record)
        groups.extend([i] * len(record))

    master_data = pd.concat(all_data, ignore_index=True)
    return master_data, groups