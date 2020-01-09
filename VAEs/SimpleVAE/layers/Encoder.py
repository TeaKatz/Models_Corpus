import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", strides=(2, 2)),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu")
        ])
        self.out1 = tf.keras.layers.Dense(latent_dim)
        self.out2 = tf.keras.layers.Dense(latent_dim)

    def call(self, inp_images):
        x = self.hidden_layers(inp_images)
        z_mean = self.out1(x)
        z_log_var = self.out2(x)
        return z_mean, z_log_var
