import tensorflow as tf


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, depth=2):
        super().__init__()
        if isinstance(filters, list):
            assert len(filters) == depth, "Length if filter exceed the layer depth."
        if isinstance(kernel_size, list):
            assert len(kernel_size) == depth, "Length if kernel_size exceed the layer depth."

        self.depth = depth
        self.filters = filters if isinstance(filters, list) else [filters for _ in range(depth)]
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size for _ in range(depth)]

        self.conv2s = [tf.keras.layers.Conv2D(self.filters[i], self.kernel_size[i], padding="same") for i in range(depth)]
        self.bns = [tf.keras.layers.BatchNormalization() for _ in range(depth)]

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
