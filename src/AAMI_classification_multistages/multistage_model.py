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
        # Build position ids
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        # Embed positions
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
    # Branch 1: short range
    conv1 = layers.Conv1D(filters, kernel_size=9, padding='same', activation='relu',)(input_tensor)
    
    # Branch 2: medium range
    conv2 = layers.Conv1D(filters, kernel_size=19, padding='same', activation='relu')(input_tensor)
    
    # Branch 3: long range
    conv3 = layers.Conv1D(filters, kernel_size=39, padding='same', activation='relu')(input_tensor)
    
    # Branch 4: pool
    pool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    conv4 = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(pool)
    
    # Merge branches
    out = layers.Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
    out = layers.BatchNormalization()(out)
    out = layers.SpatialDropout1D(0.25)(out)
    return out

def build_inception_conformer(window_size: int, n_classes: int = 18, stage: str = "multiclass") -> Model:
    # Define the input shape based on the window size
    inputs = layers.Input(shape=(window_size, 1))
    
    # Inception block
    x = inception_module(inputs, filters=32)
    x = layers.MaxPooling1D(pool_size=2)(x) 
    
    x = inception_module(x, filters=32)
    x = layers.MaxPooling1D(pool_size=2)(x) 
    
    # Dynamic sequence length for the transformer block based on the window size
    seq_length = window_size // 4
    
    # Transformer block 
    x = PositionalEmbedding(sequence_length=seq_length, output_dim=128)(x)

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.3)(x_norm, x_norm)
    x = layers.Add()([x, attention_output])

    x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn_output = layers.Dense(64, activation='relu')(x_norm2)
    ffn_output = layers.Dropout(0.3)(ffn_output)
    ffn_output = layers.Dense(128)(ffn_output)
    
    x = layers.Add()([x, ffn_output])
    
    # Classifier head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    if stage == "binary":
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Inception_Conformer")
    return model