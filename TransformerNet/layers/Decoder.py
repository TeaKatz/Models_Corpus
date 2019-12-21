import tensorflow as tf
from .DecoderLayer import DecoderLayer
from .transformer_util import positional_encoding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, maximum_position_encoding, drop_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # Word Embedding and position encoding
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        # Dropout layer before encoder layers
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        # Decoder layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, drop_rate) for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Adding embedding and position encoding
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        # Apply dropout
        x = self.dropout(x, training=training)
        # Apply Decoder layers
        for i in range(self.num_layers):
            x, attn1, attn2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights["decoder_layer{}_attention1".format(i + 1)] = attn1
            attention_weights["decoder_layer{}_attention2".format(i + 1)] = attn2
        # x.shape: (batch_size, target_seq_len, d_model)
        # attention_weight.shape: (batch_size, target_seq_len, input_seq_len)
        return x, attention_weights
