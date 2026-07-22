import tensorflow as tf
import math

@tf.function  # Compiles the function into a fast TF computational graph
def augment_ecg(signal, label):
    """
    Augments ECG signals with Gaussian noise and baseline wander.
    Parameters:
        signal (tf.Tensor): The input ECG signal tensor.
        label (tf.Tensor): The corresponding label tensor.
    Returns:
        tf.Tensor: The augmented ECG signal tensor.
        tf.Tensor: The corresponding label tensor.
    """
    # Ensure the signal is in float32 format for numerical stability
    signal = tf.cast(signal, tf.float32)
    # Clone the signal to avoid modifying the original tensor in memory
    aug_signal = tf.identity(signal)
    
    # Gaussian Noise (simulates EMG/muscle artifacts or electrode noise)
    # Apply with a 50% probability
    if tf.random.uniform(()) > 0.5:
        # stddev depends on your signal amplitude (usually between 0.01 and 0.05)
        noise = tf.random.normal(shape=tf.shape(aug_signal), mean=0.0, stddev=0.02, dtype=tf.float32)
        aug_signal = aug_signal + noise

    #Baseline Wander (simulates patient breathing)
    # Apply with a 50% probability
    if tf.random.uniform(()) > 0.5:
        window_size = tf.shape(aug_signal)[0]
        
        # Create a time vector (TensorFlow equivalent of numpy.linspace)
        t = tf.linspace(0.0, 2.0 * math.pi, window_size)
        t = tf.cast(tf.expand_dims(t, axis=-1), tf.float32) # Reshape to match the (window, 1) dimension

        # Randomize sine wave parameters: low frequency and variable amplitude
        freq = tf.random.uniform((), minval=0.1, maxval=0.4, dtype=tf.float32)
        amp = tf.random.uniform((), minval=0.05, maxval=0.2, dtype=tf.float32)
        
        baseline_wander = amp * tf.math.sin(t * freq)
        aug_signal = aug_signal + baseline_wander
    
    if tf.random.uniform(()) > 0.5:
        # Random Time Shift (Shifts the signal left or right)
        shift = tf.random.uniform(shape=[], minval=-15, maxval=15, dtype=tf.int32)
        aug_signal = tf.roll(aug_signal, shift=shift, axis=0)

    # Random Amplitude Scale (Scales the signal up or down to simulate different electrode placements or patient conditions)
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform((), minval=0.85, maxval=1.15, dtype=tf.float32)
        aug_signal = aug_signal * scale
        
    return aug_signal, label