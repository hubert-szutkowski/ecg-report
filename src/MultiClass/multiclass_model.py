import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    BatchNormalization, Activation,
    GlobalAveragePooling1D, Dropout, Dense,
    Concatenate, Add, GaussianNoise, SpatialDropout1D
)
from tensorflow.keras.regularizers import l2

def inception_module(input_tensor, filters: int):
    """
    Core Inception-1D block. 
    Processes the ECG signal through multiple kernel sizes simultaneously 
    to capture both short (high-frequency) and long (low-frequency) morphological features.
    """
    # Branch 1: Short kernel for sharp peaks (e.g., PACs, artifacts)
    conv_1 = Conv1D(filters=filters, kernel_size=10, padding='same', 
                    activation='relu', use_bias=False)(input_tensor)
    
    # Branch 2: Medium kernel for standard wave components (e.g., normal QRS)
    conv_2 = Conv1D(filters=filters, kernel_size=20, padding='same', 
                    activation='relu', use_bias=False)(input_tensor)
    
    # Branch 3: Long kernel for wide, slow waves (e.g., PVCs, T-waves)
    conv_3 = Conv1D(filters=filters, kernel_size=40, padding='same', 
                    activation='relu', use_bias=False)(input_tensor)
    
    # Branch 4: MaxPooling followed by 1x1 Conv to preserve baseline spatial data
    pool = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    conv_4 = Conv1D(filters=filters, kernel_size=1, padding='same', 
                    activation='relu', use_bias=False)(pool)
    
    # Concatenate all branches along the channel axis
    merged = Concatenate(axis=-1)([conv_1, conv_2, conv_3, conv_4])
    merged = BatchNormalization()(merged)
    merged = Activation('relu')(merged)
    
    return merged

def build_ecg_multiclass_model(input_shape: tuple, n_classes: int) -> tf.keras.Model:
    """
    Building a robust InceptionTime-1D model for multiclass ECG classification.
    
    Parameters:
        input_shape (tuple): Shape of the input data (e.g., (400, 1)).
        n_classes (int): Number of output anomaly classes.
        
    Returns:
        tf.keras.Model: Compiled Keras Functional API model.
    """
    
    # 1. Input Definition
    inputs = Input(shape=input_shape)
    
    # 2. Input Regularization (Gaussian Noise for robustness against electrode movement)
    x = GaussianNoise(0.1)(inputs)
    
    # 3. Initial Convolution (Stem) to extract low-level features before the Inception blocks
    x = Conv1D(filters=32, kernel_size=16, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(0.3)(x)
    # Store the input of the Inception block for the Residual Connection (Skip Connection)
    res_input = x 
    
    # 4. First Inception Block
    x = inception_module(x, filters=32)
    
    # 5. Skip Connection matching dimensions via 1x1 Convolution
    # This prevents the vanishing gradient problem in deep medical networks
    res_input = Conv1D(filters=x.shape[-1], kernel_size=1, padding='same', use_bias=False)(res_input)
    res_input = BatchNormalization()(res_input)
    x = Add()([x, res_input])
    x = Activation('relu')(x)
    
    # Max Pooling to compress time dimension
    x = MaxPooling1D(pool_size=4)(x)
    
    # 6. Second Inception Block (Deeper feature extraction)
    res_input = x 
    x = inception_module(x, filters=64)
    
    res_input = Conv1D(filters=x.shape[-1], kernel_size=1, padding='same', use_bias=False)(res_input)
    res_input = BatchNormalization()(res_input)
    x = Add()([x, res_input])
    x = Activation('relu')(x)

    x = SpatialDropout1D(0.2)(x)
    
    # 7. Global Average Pooling (drastically reduces parameters compared to Flatten)
    x = GlobalAveragePooling1D()(x)
    
    # 8. Fully Connected Layer with strong L2 regularization and Dropout
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.5)(x)
    
    # 9. Output Layer for Multiclass (Softmax)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    # 10. Model Compilation
    model = Model(inputs=inputs, outputs=outputs)
    
    model.summary()
    return model