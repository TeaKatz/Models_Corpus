import tensorflow as tf
from TransformerNet.layers import Encoder


def Encoder_test(*args, **kwargs):
    inputs = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)  # (batch_size, input_seq_len)

    sample_encoder = Encoder(*args, **kwargs)
    sample_encoder_output = sample_encoder(inputs, False, None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)


if __name__ == "__main__":
    Encoder_test(num_layers=2, d_model=512, num_heads=8,
                 d_ff=2048, input_vocab_size=8500,
                 maximum_position_encoding=10000)
