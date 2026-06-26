import os
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
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

    Parameters:
    dir_path (str): The path to the data directory.
    sample_select (int): The index of the record to select.

    Returns:
    pd.DataFrame: A DataFrame containing the signal and label data.
    """
    records_ids = get_record_ids(dir_path)
    record_path = str(Path(dir_path) / str(records_ids[sample_select]))

    ecg_annotations = wfdb.rdann(record_path, 'atr')
    features = ecg_annotations.__dict__['symbol']
    features_samples = ecg_annotations.__dict__['sample']
    features_np = np.array(features)
    features_A = features_samples[features_np == 'A']
    signals, _ = wfdb.rdsamp(record_path, channels=[0])
    labels = np.zeros(len(signals),dtype=int)
    labels[features_A] = 1
    record_df = pd.DataFrame({'signal': signals.flatten(), 'label': labels})

    return record_df


def data_loader(dir_path: str, selected_number_of_samples: int = 20) -> tuple:
    '''
    Loading multiple records from the data directory and combining them into a single DataFrame.
    Parameters:
    dir_path (str): The path to the data directory.
    selected_number_of_samples (int): The number of records to load.
    Returns:
    tuple: A tuple containing the combined DataFrame and a list of group labels.
    '''
    
    records_list = []
    for i in range(selected_number_of_samples):
        record_df = get_record(dir_path, sample_select=i)
        records_list.append(record_df)
    all_data = []
    groups = []
    for i, record in enumerate(records_list):
        all_data.append(record)
        groups.extend([i]*len(record))
    master_data = pd.concat(all_data, ignore_index=True)
    return master_data, groups