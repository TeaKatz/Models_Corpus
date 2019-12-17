import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Layer


class IdentityBlock(Layer):
    def __init__(self, filters, kernel_size, depth=2):
        super().__init__()
        if isinstance(filters, list):
            assert len(filters) == depth, "Length if filter exceed the layer depth."
        if isinstance(kernel_size, list):
            assert len(kernel_size) == depth, "Length if kernel_size exceed the layer depth."

        self.depth = depth
        self.filters = filters if isinstance(filters, list) else [filters for _ in range(depth)]
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size for _ in range(depth)]

        self.conv2s = [Conv2D(self.filters[i], self.kernel_size[i], padding="same") for i in range(depth)]
        self.bns = [BatchNormalization() for _ in range(depth)]

    def call(self, input_tensor, training=False):
        x = input_tensor
        for i in range(len(self.conv2s) - 1):
            x = self.conv2s[i](x)
            x = self.bns[i](x, training=training)
            x = tf.nn.relu(x)

        x = self.conv2s[-1](x)
        x = self.bns[-1](x, training=training)

        x += input_tensor
        x = tf.nn.relu(x)

        return x


class ProjectionBlock(Layer):
    def __init__(self, filters, kernel_size, depth=2, strides=2):
        super().__init__()
        if isinstance(filters, list):
            assert len(filters) == depth, "Length if filter exceed the layer depth."
        if isinstance(kernel_size, list):
            assert len(kernel_size) == depth, "Length if kernel_size exceed the layer depth."

        self.depth = depth
        self.filters = filters if isinstance(filters, list) else [filters for _ in range(depth)]
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size for _ in range(depth)]

        self.conv2s = [Conv2D(self.filters[i], self.kernel_size[i], strides=strides, padding="same") if i == 0 else
                       Conv2D(self.filters[i], self.kernel_size[i], padding="same") for i in range(depth)]
        self.bns = [BatchNormalization() for _ in range(depth)]

        self.conv2_id = Conv2D(self.filters[-1], 1, strides=strides, padding="same")
        self.bn_id = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = input_tensor
        for i in range(len(self.conv2s) - 1):
            x = self.conv2s[i](x)
            x = self.bns[i](x)
            x = tf.nn.relu(x)

        x = self.conv2s[-1](x)
        x = self.bns[-1](x, training=training)

        id = self.conv2_id(input_tensor)
        id = self.bn_id(id)

        x += id
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
