import numpy as np
import tensorflow as tf


def positional_encoding(position, d_model):
    """ Since TransformerNet so not has position bias like RNN or CNN, without a positional encoding it will has no idea of word position matter """
    def get_angles(position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates  # (position, d_model)

    angle_rads = get_angles(position=np.arange(position)[:, np.newaxis],
                            i=np.arange(d_model)[np.newaxis, :],
                            d_model=d_model)

    # Apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)  # (1, position, d_model)


def create_padding_mask(seq):
    """ Create mask to inform position of padding to model to avoid model treats padding as input """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(seq_len):
    """ Create mask for decoder to inform it which word should not be used at each step """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)   # (..., seq_len_q, seq_len_k)

    # Scale matmul_qk by square root of depth
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (..., seq_len_q, seq_len_k)

    # Add mask to scaled tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax on the last axis (seq_len_k) to normalize so it add up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)     # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)    # (..., seq_len_q, depth_v)
    return output, attention_weights  # (..., seq_len_q, depth_v), (..., seq_len_q, seq_len_k)


def point_wise_feed_forward_network(d_model, dff):
    """
    Fully connected block after Multi-Head Attention layer
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
    return model
