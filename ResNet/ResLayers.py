import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Layer


class IdentityBlock(Layer):
    def __init__(self, filters, kernel_size, depth=2):
        super().__init__()
        self.depth = depth
        self.conv2s = [Conv2D(filters, kernel_size, padding="same") for _ in range(depth - 1)]
        self.bns = [BatchNormalization() for _ in range(depth - 1)]
        self.conv2_last = Conv2D(filters, kernel_size, padding="same")
        self.bn_last = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = input_tensor
        for i in range(self.depth - 1):
            x = self.conv2s[i](x)
            x = self.bns[i](x, training=training)
            x = tf.nn.relu(x)

        x = self.conv2_last(x)
        x = self.bn_last(x, training=training)

        x += input_tensor
        x = tf.nn.relu(x)

        return x


class ProjectionBlock(Layer):
    def __init__(self, filters, kernel_size, depth=2):
        super().__init__()
        self.depth = depth
        self.conv2_first = Conv2D(filters, kernel_size, strides=2, padding="same")
        self.bn_first = BatchNormalization()
        self.conv2s = [Conv2D(filters, kernel_size, padding="same") for _ in range(depth - 2)]
        self.bns = [BatchNormalization() for _ in range(depth - 2)]
        self.conv2_last = Conv2D(filters, kernel_size, padding="same")
        self.bn_last = BatchNormalization()
        self.conv2_id = Conv2D(filters, 1, strides=2, padding="same")

    def call(self, input_tensor, training=False):
        x = self.conv2_first(input_tensor)
        x = self.bn_first(x)

        for i in range(self.depth - 2):
            x = self.conv2s[i](x)
            x = self.bns[i](x, training=training)
            x = tf.nn.relu(x)

        x = self.conv2_last(x)
        x = self.bn_last(x, training=training)

        x += self.conv2_id(input_tensor)
        x = tf.nn.relu(x)

        return x


if __name__ == "__main__":
    import numpy as np

    idBlock = IdentityBlock(64, 3)
    pjBlock = ProjectionBlock(128, 3)

    input_tensor = np.random.rand(10, 32, 32, 64)
    print("input tensor shape: {}".format(input_tensor.shape))
    idBlock_output = idBlock(input_tensor)
    print("idBlock output shape: {} (pass={})".format(idBlock_output.shape, idBlock_output.shape == input_tensor.shape))
    pjBlock_output = pjBlock(input_tensor)
    print("pjBlock output shape: {} (pass={})".format(pjBlock_output.shape, pjBlock_output.shape == (10, 16, 16, 128)))
