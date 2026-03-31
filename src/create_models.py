#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, GRU, Input
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras_nlp.layers import SinePositionEncoding, TransformerEncoder

# compile model using accuracy to measure model performance
# modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# CONVOLUTIONALS----------------------------------
def create_cnn_v1_0():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(20, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(Conv2D(32, kernel_size=(1, 1), activation="relu"))
    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_0"

def create_cnn_v1_1():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(13, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_1"

def create_cnn_v1_2():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(9, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(3, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_2"

def create_cnn_v1_3():
    modelo = Sequential()
    modelo.add(Conv2D(64, kernel_size=(5, 1), activation="relu", input_shape=(20, 12, 1)))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Conv2D(32, kernel_size=(2, 1), activation="relu"))
    modelo.add(MaxPooling2D((2, 1)))

    modelo.add(Flatten())
    modelo.add(Dense(4, activation="softmax"))
    return modelo, "cnn_v1_3"

# RECURRENTS----------------------------------------

def create_lstm_v1_0():
    model = Sequential()
    model.add(LSTM(16, input_shape=(20, 12)))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_0"


def create_lstm_v1_1():
    model = Sequential()
    model.add(LSTM(16, input_shape=(20, 12), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_1"


def create_lstm_v1_2():
    model = Sequential()
    model.add(LSTM(16, input_shape=(20, 12), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_2"


def create_lstm_v1_3():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 12)))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_3"


def create_lstm_v1_4():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 12), return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_4"


def create_lstm_v1_5():
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 12), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(4, activation="softmax"))
    return model, "lstm_v1_5"


# TRANSFORMERS------------------------------------------------

def create_transformer_v1_0():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers = 1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_0"


def create_transformer_v1_1():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=8, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_1"


def create_transformer_v1_2():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_2"


def create_transformer_v1_3():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=8, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_3"


def create_transformer_v1_4():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=2
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_4"


def create_transformer_v1_5():
    inputs = keras.Input(shape=(20,12))
    positional_encoding = SinePositionEncoding()(inputs)
    x = inputs + positional_encoding
    num_layers=2
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=8, activation="relu", intermediate_dim=2048)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model, "transformer_v1_5"


class RelativePositionEncoding(layers.Layer):
    def __init__(self, max_len, **kwargs):
        super(RelativePositionEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        # Create the positional encodings based on relative position indices
        self.position_embeddings = self.add_weight(
            shape=(max_len, max_len),  # Shape is (seq_len, seq_len) for relative positions
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]  # Get batch size (None)
        seq_len = tf.shape(inputs)[1]  # Get sequence length (20 in your case)

        # Create a position matrix (relative distances between positions)
        pos = tf.expand_dims(tf.range(seq_len), 0) - tf.expand_dims(tf.range(seq_len), 1)

        # Map the relative positions to the embeddings using the learned weights
        relative_pos_embeddings = tf.gather(self.position_embeddings, pos + self.max_len // 2)

        # Reshape the relative positional encodings to have shape (batch_size, seq_len, seq_len)
        relative_pos_embeddings = tf.expand_dims(relative_pos_embeddings, 0)  # Add batch dimension
        relative_pos_embeddings = tf.tile(relative_pos_embeddings, [batch_size, 1, 1])  # Broadcast to match batch size

        # Expand the relative_pos_embeddings to match the feature size dimension of the input
        relative_pos_embeddings = tf.expand_dims(relative_pos_embeddings, -1)  # Add feature dimension
        relative_pos_embeddings = tf.tile(relative_pos_embeddings,
                                          [1, 1, 1, tf.shape(inputs)[-1]])  # Broadcast across features

        # Add the relative positional encodings to the input embeddings (real values)
        return inputs + relative_pos_embeddings

    def compute_output_shape(self, input_shape):
        # Output shape is the same as the input shape since we're just adding embeddings
        return input_shape


def create_transformer_v3_0():
    inputs = layers.Input(shape=(20, 12))

    # Apply relative positional encoding to the real-valued input
    x = RelativePositionEncoding(max_len=20)(inputs)

    num_layers = 1
    # Transformer Encoder Layers
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=4, activation="relu", intermediate_dim=512)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(4, activation="softmax")(x)  # 4-class classification
    model = keras.Model(inputs, outputs)

    return model
