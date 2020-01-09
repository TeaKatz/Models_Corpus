import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def call(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=(z_mean.shape[0], self.latent_dim), mean=0.0, stddev=1.0)
        z = z_mean + tf.exp(z_log_var) * epsilon
        return z
