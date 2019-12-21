from TransformerNet.layers.transformer_util import *


def positional_encoding_test(*args, **kwargs):
    pos_encoding = positional_encoding(*args, **kwargs)
    print(pos_encoding.shape)
    print(pos_encoding)
    print("------------------------------------------")


def create_padding_mask_test(*args, **kwargs):
    mask = create_padding_mask(*args, **kwargs)
    print(mask.shape)
    print(mask)
    print("------------------------------------------")


def create_look_ahead_mask_test(*args, **kwargs):
    mask = create_look_ahead_mask(*args, **kwargs)
    print(mask.shape)
    print(mask)
    print("------------------------------------------")


def scaled_dot_product_attention_test(*args, **kwargs):
    def print_out(q, k, v):
        temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
        print("Attention weights are:")
        print(temp_attn)
        print("Output is:")
        print(temp_out)

    output, attention_weights = scaled_dot_product_attention(*args, **kwargs)
    print(output.shape)
    print(attention_weights.shape)
    print_out(*args, **kwargs)
    print("------------------------------------------")


def point_wise_feed_forward_network_test(*args, **kwargs):
    sample_ffn = point_wise_feed_forward_network(*args, **kwargs)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
    print("------------------------------------------")


if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    np.set_printoptions(suppress=True)
    ###################################################################################
    positional_encoding_test(50, 512)
    ###################################################################################
    create_padding_mask_test([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    create_padding_mask_test(np.array([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]))
    create_padding_mask_test(tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]))
    create_padding_mask_test(tf.convert_to_tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]))
    ###################################################################################
    create_look_ahead_mask_test(np.ones((3, 3)).shape[1])
    create_look_ahead_mask_test(tf.ones((3, 3)).shape[1])
    ###################################################################################
    # (4, 3)
    temp_k = tf.constant([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10],
                          [0, 0, 10]], dtype=tf.float32)
    # (4, 2)
    temp_v = tf.constant([[1, 0],
                          [10, 0],
                          [100, 5],
                          [1000, 6]], dtype=tf.float32)
    # (1, 3)
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
    scaled_dot_product_attention_test(temp_q, temp_k, temp_v)
    # (1, 3)
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
    scaled_dot_product_attention_test(temp_q, temp_k, temp_v)
    # (1, 3)
    temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)
    scaled_dot_product_attention_test(temp_q, temp_k, temp_v)
    # (3, 3)
    temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)
    scaled_dot_product_attention_test(temp_q, temp_k, temp_v)
    ###################################################################################
    point_wise_feed_forward_network_test(512, 2048)
