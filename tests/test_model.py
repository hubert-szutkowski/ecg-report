import sys
import os
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from MultiClass.multiclass_model import build_ecg_multiclass_model

def test_build_ecg_multiclass_model_architecture():
    """
    Verify that the model compiles with correct input/output shapes 
    and uses the mandatory softmax activation for multiclass classification.
    """
    input_shape = (400, 1)
    n_classes = 6
    
    # Model instance
    model = build_ecg_multiclass_model(input_shape=input_shape, n_classes=n_classes)
    
    # 1. Assert input shape 
    assert model.input_shape == (None, 400, 1), f"Expected input shape (None, 400, 1), got {model.input_shape}"
    
    # 2. Assert output shape
    assert model.output_shape == (None, 6), f"Expected output shape (None, 6), got {model.output_shape}"
    
    # Check softmax output
    final_layer_activation = model.layers[-1].activation.__name__
    assert final_layer_activation == 'softmax', f"Expected 'softmax' activation, got '{final_layer_activation}'"