import tensorflow as tf
from TransformerNet.layers import EncoderLayer, DecoderLayer


def DecoderLayer_test(*args, **kwargs):
    inputs = tf.random.uniform((64, 43, 512))  # (batch_size, input_seq_len, word_vector)
    enc_output = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)(inputs, False, None)  # (batch_size, input_seq_len, d_model)
    target = tf.random.uniform((64, 50, 512))  # (batch_size, target_seq_len, word_vector)

    sample_decoder_layer = DecoderLayer(*args, **kwargs)
    sample_decoder_layer_output, _, _ = sample_decoder_layer(target, enc_output, False, None, None)  # (batch_size, target_seq_len, d_model)
    print(sample_decoder_layer_output.shape)


if __name__ == "__main__":
    DecoderLayer_test(d_model=512, num_heads=8, d_ff=2048)
