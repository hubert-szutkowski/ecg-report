import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    BatchNormalization, Activation,
    GlobalAveragePooling1D, Dropout, Dense
)
from tensorflow.keras.regularizers import l2


def build_ecg_model(input_shape: tuple, n_classes: int) -> tf.keras.Model:
    """
    Building a 1D CNN model for ECG classification.
    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (4096, 1)).
        n_classes (int): Number of output classes (1 for binary classification).
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """

    def conv_block(filters: int, kernel_size: int):
        return [
            Conv1D(filters, kernel_size=kernel_size, padding='same', use_bias=False),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(4),
            Dropout(0.3),
        ]

    model = Sequential([
        Input(shape=input_shape),           # (4096, 1)
        *conv_block(32,  kernel_size=16),   # → (1024, 32)
        *conv_block(64,  kernel_size=8),    # → (256,  64)
        *conv_block(128, kernel_size=4),    # → (64,   128)

        GlobalAveragePooling1D(),           # → (128,)

        Dense(64, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.5),

        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )

    model.summary()
    return model