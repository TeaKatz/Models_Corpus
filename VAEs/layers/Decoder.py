import numpy as np
import tensorflow as tf


class Decoder(tf.keras.layers.Layer):
    def __init__(self, shape_before_flatten):
        super().__init__()
        self.hidden_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(np.prod(shape_before_flatten), activation="relu"),
            tf.keras.layers.Reshape(shape_before_flatten),
            tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu", strides=(2, 2)),
            tf.keras.layers.Conv2DTranspose(32, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2DTranspose(1, 3, padding="same", activation="sigmoid")
        ])

    def call(self, z_sampling):
        decoded_images = self.hidden_layers(z_sampling)
        return decoded_images
