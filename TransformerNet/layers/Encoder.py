import tensorflow as tf
from .EncoderLayer import EncoderLayer
from .transformer_util import positional_encoding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, drop_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # Word Embedding and position encoding
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        # Dropout layer before encoder layers
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        # Encoder layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, drop_rate) for _ in range(num_layers)]

    def call(self, x, training, padding_mask):
        seq_len = tf.shape(x)[1]

        # Adding embedding and position encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        # Apply dropout
        x = self.dropout(x, training=training)
        # Apply Encoder layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, padding_mask)
        return x  # (batch_size, input_seq_len, d_model)
