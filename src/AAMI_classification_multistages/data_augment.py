import tensorflow as tf
import math

@tf.function  # Fast TF graph
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
    # Cast to float32
    signal = tf.cast(signal, tf.float32)
    # Copy input
    aug_signal = tf.identity(signal)
    
    # Gaussian noise
    if tf.random.uniform(()) > 0.5:
        # Small noise level
        noise = tf.random.normal(shape=tf.shape(aug_signal), mean=0.0, stddev=0.02, dtype=tf.float32)
        aug_signal = aug_signal + noise

    # Baseline wander
    if tf.random.uniform(()) > 0.5:
        window_size = tf.shape(aug_signal)[0]
        
        # Build time axis
        t = tf.linspace(0.0, 2.0 * math.pi, window_size)
        t = tf.cast(tf.expand_dims(t, axis=-1), tf.float32) # Match (window, 1)

        # Random sine parameters
        freq = tf.random.uniform((), minval=0.1, maxval=0.4, dtype=tf.float32)
        amp = tf.random.uniform((), minval=0.05, maxval=0.2, dtype=tf.float32)
        
        baseline_wander = amp * tf.math.sin(t * freq)
        aug_signal = aug_signal + baseline_wander
    
    if tf.random.uniform(()) > 0.5:
        # Random shift
        shift = tf.random.uniform(shape=[], minval=-15, maxval=15, dtype=tf.int32)
        aug_signal = tf.roll(aug_signal, shift=shift, axis=0)

    # Random scale
    if tf.random.uniform(()) > 0.5:
        scale = tf.random.uniform((), minval=0.85, maxval=1.15, dtype=tf.float32)
        aug_signal = aug_signal * scale
        
    return aug_signal, label