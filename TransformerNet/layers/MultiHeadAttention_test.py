import tensorflow as tf
from TransformerNet.layers import MultiHeadAttention


def MultiHeadAttention_test(*args, **kwargs):
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)

    temp_mha = MultiHeadAttention(*args, **kwargs)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print(out.shape)
    print(attn.shape)


if __name__ == "__main__":
    MultiHeadAttention_test(d_model=512, num_heads=8)
