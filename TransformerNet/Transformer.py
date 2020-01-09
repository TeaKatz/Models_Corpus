import tensorflow as tf
from TransformerNet.layers import Encoder, Decoder


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, target_vocab_size, pe_input, pe_target, drop_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, pe_input, drop_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size, pe_target, drop_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, input_seq_len, d_model)
        # dec_output.shape: (batch_size, target_seq_len, d_model)
        # attention_weight.shape: (batch, target_seq_len, input_seq_len)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, target_seq_len, target_vocab_size)
        return final_output, attention_weights
