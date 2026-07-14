import sys
import os
import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from MultiClass.multiclass_preprocessing import extract_anomaly_windows

def test_extract_anomaly_windows_filtering_and_shapes():
    """
    Test if normal beats are properly skipped, edge cases are protected,
    and the extracted windows have the exact expected length of 400 samples.
    """
    # Mock signal
    signals = np.sin(np.linspace(0, 50, 1000))
    
    # Mock annotations:
    # - Index 200: Anomaly 'V' (Valid, should be extracted)
    # - Index 500: Normal beat 'N' (Should be skipped)
    # - Index 900: Anomaly 'A' (Out of bounds: 900 + 250 > 1000, should be skipped)
    features_samples = [200, 500, 900]
    features = ['V', 'N', 'A']
    
    window_back = 150
    window_forward = 250
    
    # Execute the function
    X, y = extract_anomaly_windows(
        signals=signals,
        features_samples=features_samples,
        features=features,
        window_back=window_back,
        window_forward=window_forward
    )
    
    # Assertions
    # Out of 3 peaks, only the first one ('V' at index 200) satisfies all conditions
    assert len(X) == 1, f"Expected 1 extracted window, got {len(X)}"
    assert len(y) == 1, f"Expected 1 label, got {len(y)}"
    assert y[0] == 'V', f"Expected label to be 'V', got '{y[0]}'"
    
    # Check if the window size is exactly 400 (150 back + 250 forward)
    assert X.shape[1] == 400, f"Expected window width of 400, got {X.shape[1]}"
    
    # Verify that the extracted window matches the exact sliced mathematical signal
    expected_slice = signals[200 - window_back : 200 + window_forward].flatten()
    np.testing.assert_array_equal(X[0], expected_slice)


def test_extract_anomaly_windows_empty_inputs():
    """
    Test how the function handles edge cases with empty input data arrays.
    """
    signals = np.array([])
    features_samples = []
    features = []
    
    X, y = extract_anomaly_windows(signals, features_samples, features)
    
    assert len(X) == 0
    assert len(y) == 0