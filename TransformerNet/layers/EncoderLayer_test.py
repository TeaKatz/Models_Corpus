import tensorflow as tf
from TransformerNet.layers import EncoderLayer


def EncoderLayer_test(*args, **kwargs):
    inputs = tf.random.uniform((64, 43, 512))  # (batch_size, seq_len, word_vector)

    sample_encoder_layer = EncoderLayer(*args, **kwargs)
    sample_encoder_layer_output = sample_encoder_layer(inputs, False, None)  # (batch_size, seq_len, d_model)
    print(sample_encoder_layer_output.shape)


if __name__ == "__main__":
    EncoderLayer_test(d_model=512, num_heads=8, d_ff=2048)
