import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    BatchNormalization, Activation,
    GlobalAveragePooling1D, Dropout, Dense,
    SpatialDropout1D
    , GaussianNoise  
)
from tensorflow.keras.losses import BinaryFocalCrossentropy  
from tensorflow.keras.regularizers import l2

def build_ecg_model(input_shape: tuple, n_classes: int) -> tf.keras.Model:
    """
    Building a robust 1D CNN model for ECG classification.
    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (1024, 1)).
        n_classes (int): Number of output classes (1 for binary classification).
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """

    def conv_block(filters: int, kernel_size: int):
        return [
            # 1. Lekkie L2 w konwolucjach zapobiega drastycznym wagom reagującym na szum
            Conv1D(filters, kernel_size=kernel_size, padding='same', use_bias=False, 
                   kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(4),
            # 2. Zmiana na SpatialDropout1D - wyłącza całe mapy cech, a nie pojedyncze punkty
            SpatialDropout1D(0.3),
        ]

    model = Sequential([
        Input(shape=input_shape),           
        
        # 3. Szum Gaussa na wejściu. Zakładamy, że sygnał przeszedł przez StandardScaler, 
        # więc odchylenie 0.05 to 5% wariancji. Symuluje to realne zakłócenia z elektrod.
        GaussianNoise(0.05),

        *conv_block(32,  kernel_size=16),   
        *conv_block(64,  kernel_size=8),    
        *conv_block(128, kernel_size=4),    

        GlobalAveragePooling1D(),           

        # Mocniejsza regularyzacja na warstwie w pełni połączonej
        Dense(64, activation='relu', kernel_regularizer=l2(1e-3)),
        Dropout(0.5),

        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=BinaryFocalCrossentropy(gamma=2.0),  
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )

    model.summary()
    return model