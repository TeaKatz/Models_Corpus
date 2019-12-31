import tensorflow as tf


class CustomLossFunction:
    @staticmethod
    def reconstruction_loss(y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        loss = tf.keras.metrics.binary_crossentropy(y_true, y_pred)
        return loss

    @staticmethod
    def regularization_loss(z_mean, z_log_var):
        loss = -5e-4 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        return loss

    def vae_loss(self, y_true, y_pred, z_mean, z_log_var):
        rec_loss = self.reconstruction_loss(y_true, y_pred)
        reg_loss = self.regularization_loss(z_mean, z_log_var)
        loss = tf.reduce_mean(rec_loss + reg_loss)
        return loss

    def __call__(self, y_true, y_pred, z_mean, z_log_var):
        loss = self.vae_loss(y_true, y_pred, z_mean, z_log_var)
        return loss
