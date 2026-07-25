import sys
import os
import numpy as np
import pytest

def test_groups_alignment_for_cross_validation():
    """
    Ensure that the generated groups array aligns perfectly with data samples,
    which is a hard requirement for StratifiedGroupKFold.
    """
    # Simulate data aggregation 
    record_106_windows = np.random.rand(15, 400) # 15 windows from patient 106
    record_119_windows = np.random.rand(10, 400) # 10 windows from patient 119
    
    # Match aggregation logic
    X_list = [record_106_windows, record_119_windows]
    record_ids = ['106', '119']
    
    groups_all = []
    for i, X_record in enumerate(X_list):
        groups_all.extend([record_ids[i]] * len(X_record))
        
    X_master = np.vstack(X_list)
    groups_master = np.array(groups_all)
    
    # Assertions
    assert len(X_master) == len(groups_master), "Length mismatch between features and cross-validation groups!"
    assert len(groups_master) == 25, f"Expected total 25 group elements, got {len(groups_master)}"
    assert np.sum(groups_master == '106') == 15, "Incorrect count for group '106'"
    assert np.sum(groups_master == '119') == 10, "Incorrect count for group '119'"