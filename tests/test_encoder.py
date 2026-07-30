import sys
import os
import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def test_label_encoding_and_one_hot_conversion():
    """
    Ensure that string labels are correctly transformed into integers 
    and then successfully converted to one-hot encoded matrices.
    """
    # Mock extracted anomaly labels
    mock_y_labels = np.array(['V', 'A', 'V', 'R', 'A', 'V'])
    
    # Initialize and fit encoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(mock_y_labels)
    
    # Assert unique classes
    assert len(encoder.classes_) == 3
    assert set(encoder.classes_) == {'A', 'R', 'V'}
    
    # Perform one-hot encoding
    num_classes = len(encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    # Check output shape
    assert y_categorical.shape == (6, 3), f"Expected shape (6, 3), got {y_categorical.shape}"
    
    # Check row sums
    row_sums = np.sum(y_categorical, axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(6))