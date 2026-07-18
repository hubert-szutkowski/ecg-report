import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class PositionalEmbedding(layers.Layer):
    """
    Layer that adds positional embeddings to the input tensor. This is useful for models that need to capture the order of elements in a sequence, such as transformers.
    Parameters:
        sequence_length (int): The length of the input sequences.
        output_dim (int): The dimensionality of the positional embeddings.
    Returns:
        tf.Tensor: The input tensor with added positional embeddings.
    """
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length

    def call(self, inputs):
        # Generating a range of positions for the input sequence
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        # Changing the shape of positions to match the input tensor's batch size
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions


def inception_module(input_tensor, filters=32):
    """
    Inception module for 1D signals. It consists of multiple convolutional branches with different kernel sizes and a pooling branch, which are then concatenated together.
    Parameters:
        input_tensor (tf.Tensor): Input tensor to the inception module.
        filters (int): Number of filters for each convolutional branch.
    Returns:
        tf.Tensor: Output tensor after applying the inception module.
    """
    #Branch 1: short signals (e.g., P wave)
    conv1 = layers.Conv1D(filters, kernel_size=9, padding='same', activation='relu',)(input_tensor)
    
    # Branch 2: Medium signals (e.g., QRS complex)
    conv2 = layers.Conv1D(filters, kernel_size=19, padding='same', activation='relu')(input_tensor)
    
    # Branch 3: Long signals (e.g., T or U wave)
    conv3 = layers.Conv1D(filters, kernel_size=39, padding='same', activation='relu')(input_tensor)
    
    # Branch 4: Pooling
    pool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    conv4 = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(pool)
    
    # Merging all branches
    out = layers.Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
    out = layers.BatchNormalization()(out)
    out = layers.SpatialDropout1D(0.25)(out)
    return out

def build_inception_conformer(input_shape=(400, 1), n_classes=18):
    inputs = layers.Input(shape=input_shape)
    
    
    #INCEPTION (Morphological feature extraction)
    
    x = inception_module(inputs, filters=32)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduction to 200 samples
    
    x = inception_module(x, filters=32)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduction to 100 samples
    
    # TRANSFORMER (Time-series modeling)
    
    attention_output = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=64, 
        dropout=0.35
    )(x, x)
    
    # Residual connection and layer normalization
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward network inside the transformer block
    ffn_output = layers.Dense(64, activation='relu')(x)
    ffn_output = layers.Dropout(0.3)(ffn_output)
    ffn_output = layers.Dense(128)(ffn_output)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Inception_Conformer")
    return model