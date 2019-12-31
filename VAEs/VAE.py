import tensorflow as tf
from VAEs.layers import Decoder, Encoder, Sampling


class VAE(tf.keras.Model):
    def __init__(self, shape_before_flattening, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.decoder = Decoder(shape_before_flattening)

    def call(self, inp):
        z_mean, z_log_var = self.encoder(inp)
        z = self.sampling(z_mean, z_log_var)
        y_pred = self.decoder(z)
        return y_pred, z_mean, z_log_var
