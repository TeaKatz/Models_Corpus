from VAEs.utils import CustomModel
from VAEs.SimpleVAE.layers import Decoder, Encoder, Sampling


class SimpleVAE(CustomModel):
    def __init__(self, shape_before_flattening, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling(latent_dim)
        self.decoder = Decoder(shape_before_flattening)

    def call(self, inp):
        z_mean, z_log_var = self.encoder(inp)
        z = self.sampling(z_mean, z_log_var)
        y_pred = self.decoder(z)
        return y_pred, z_mean, z_log_var
