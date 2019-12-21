import tensorflow as tf
from .MultiHeadAttention import MultiHeadAttention
from .transformer_util import point_wise_feed_forward_network


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, drop_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)

        # Normalization on d_model (last axis)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, axis=-1)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, padding_mask):
        attn_output, _ = self.mha(x, x, x, padding_mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2
