import tensorflow as tf
from ResNet.layers import IdentityBlock, ProjectionBlock


class ResNet34(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv2 = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same")
        self.maxpool = tf.keras.layers.MaxPool2D(3, strides=2, padding="valid")
        self.idblk_1 = [IdentityBlock(64, 3) for _ in range(3)]
        self.pjblk_1 = ProjectionBlock(128, 3)
        self.idblk_2 = [IdentityBlock(128, 3) for _ in range(3)]
        self.pjblk_2 = ProjectionBlock(256, 3)
        self.idblk_3 = [IdentityBlock(256, 3) for _ in range(5)]
        self.pjblk_3 = ProjectionBlock(512, 3)
        self.idblk_4 = [IdentityBlock(512, 3) for _ in range(2)]
        self.avgpool = tf.keras.layers.AvgPool2D(3, strides=2, padding="valid")

    def call(self, input_tensor, training=False):
        assert input_tensor.shape[1] >= 112 and input_tensor.shape[2] >= 112, \
            "Input tensor should have width and heigth at least 112"

        # (112, 112, 3) -> (56, 56, 64)
        x = self.conv2(input_tensor)
        # (56, 56, 64) -> (27, 27, 64)
        x = self.maxpool(x)
        for layer in self.idblk_1:
            x = layer(x, training=training)
        # (27, 27, 64) -> (14, 14, 128)
        x = self.pjblk_1(x, training=training)
        for layer in self.idblk_2:
            x = layer(x, training=training)
        # (14, 14, 128) -> (7, 7, 256)
        x = self.pjblk_2(x, training=training)
        for layer in self.idblk_3:
            x = layer(x, training=training)
        # (7, 7, 256) -> (4, 4, 512)
        x = self.pjblk_3(x, training=training)
        for layer in self.idblk_4:
            x = layer(x, training=training)
        # (4, 4, 512) -> (1, 1, 512)
        x = self.avgpool(x)

        return x


if __name__ == "__main__":
    import numpy as np

    resnet = ResNet34()
    resnet.build(input_shape=(1, 112, 112, 3))
    resnet.summary()

    input_tensor = np.random.rand(10, 112, 112, 3)
    print("input tensor shape: {}".format(input_tensor.shape))
    output_tensor = resnet(input_tensor)
    print("output tensor shape: {}".format(output_tensor.shape))
