import tensorflow as tf
from TransformerNet import Transformer


def Transformer_test(*args, **kwargs):
    inputs = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)  # (batch_size, input_seq_len)
    target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)  # (batch)size, target_seq_len)

    sample_transformer = Transformer(*args, **kwargs)
    fn_out, _ = sample_transformer(inputs, target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, target_seq_len, target_vocab_size)


if __name__ == "__main__":
    Transformer_test(num_layers=2, d_model=512, num_heads=8, d_ff=2048,
                     input_vocab_size=8500, target_vocab_size=8000,
                     pe_input=10000, pe_target=6000)
