import numpy as np
import tensorflow as tf


class BaggingClassifier(tf.keras.Model):
    def __init__(self, base_learner, model_nums=2):
        super().__init__()
        self.meta_learners = [tf.keras.models.clone_model(base_learner) for _ in range(model_nums)]
        self.model_nums = model_nums
        self.hostory = {}

    def call(self, inp, training=False, soft_voting=True):
        # Get prediction from each model
        y_preds = []
        for model in self.meta_learners:
            y_pred = model(inp)
            y_preds.append(y_pred)

        # Voting
        if soft_voting:
            # Soft voting
            y_mean = np.mean(y_preds, axis=0)
            return np.argmax(y_mean)
        else:
            # Hard voting
            y_max = np.argmax(y_preds, axis=1)
            return np.argmax(np.bincount(y_max))

    def fit(self, x, y, batch_size=32, epochs=10, verbose=100):
        assert tf.is_tensor(x) or isinstance(x, np.ndarray), "x must be either tensor or numpy array"
        assert tf.is_tensor(y) or isinstance(y, np.ndarray), "y must be either tensor or numpy array"

        # Create indices for bootstrapping
        indices = np.random.randint(0, x.shape[0], size=[self.model_nums, x.shape[0]])

        # Train each model
        self.hostory = {}
        for i, model in enumerate(self.meta_learners):
            # Sampling training data
            sampling_x = x[indices[i]]
            sampling_y = y[indices[i]]
            # Train
            model.fit(sampling_x, sampling_y, batch_size=batch_size, epochs=epochs, verbose=verbose)
