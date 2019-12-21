import tensorflow as tf
import matplotlib.pyplot as plt
from TransformerNet.utils import CustomSchedule


def CustomSchedule_test(*args, **kwargs):
    learning_rate = CustomSchedule(*args, **kwargs)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()


if __name__ == "__main__":
    CustomSchedule_test(128)
