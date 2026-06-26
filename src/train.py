from preprocessing import  get_record_ids, data_loader
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
#Get path to the data directory
base_dir = Path(__file__).resolve().parent.parent
dir_path = base_dir / 'data'
dir_path_str = str(dir_path)
dir_path_str = str(dir_path)
record_ids = get_record_ids(dir_path_str)
#Get record
TOTAL_NUMBER_OF_SAMPLES = len(record_ids)
SELECTED_NUMBER_OF_SAMPLES = 20

MASTER_DATA, GROUPS = data_loader(dir_path_str, selected_number_of_samples=SELECTED_NUMBER_OF_SAMPLES)

SIGNAL = MASTER_DATA[['signal']]
LABELS = MASTER_DATA['label']
GROUPS = np.array(GROUPS)
FOLD_SPLITS = 5

print(f"Rozmiar sygnału: {len(SIGNAL)}")
print(f"Rozmiar etykiet: {len(LABELS)}")
print(f"Rozmiar grup:    {len(GROUPS)}")

train_test_split_dict = {}
fold_number = 0
for train_index, test_index in GroupKFold(n_splits=FOLD_SPLITS).split(SIGNAL, LABELS, GROUPS):
    X_train, X_test = SIGNAL.iloc[train_index], SIGNAL.iloc[test_index]
    y_train, y_test = LABELS.iloc[train_index], LABELS.iloc[test_index]
    group_train, group_test = GROUPS[train_index], GROUPS[test_index]
    print(f"Trening - Wierszy: {len(X_train)}, Unikalne grupy (df-y): {np.unique(group_train)}")
    print(f"Test    - Wierszy: {len(X_test)}, Unikalne grupy (df-y): {np.unique(group_test)}")
    train_test_split_dict[fold_number] = (np.unique(group_train), np.unique(group_test))
    fold_number += 1


print(train_test_split_dict)