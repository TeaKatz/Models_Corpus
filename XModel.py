import time
import numpy as np
import tensorflow as tf


class XModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loss = None
        self.train_metrics = None
        self.val_loss = None
        self.val_metrics = None
        self.test_loss = None
        self.test_metrics = None
        self.is_compiled = False

    def compile(self, **kwargs):
        super().compile(**kwargs)
        # Ser flag
        self.is_compiled = True
        # Define average for loss
        if self.loss:
            self.train_loss = tf.metrics.Mean()
            self.val_loss = tf.metrics.Mean()
            self.test_loss = tf.metrics.Mean()
        # Define average for metrics
        if self._compile_metrics:
            self.train_metrics = tf.metrics.Mean()
            self.val_metrics = tf.metrics.Mean()
            self.test_metrics = tf.metrics.Mean()

    def _train_step(self, x, y):
        # Forward propagation
        with tf.GradientTape() as g:
            g.watch(x)
            y_pred = self(x)
            loss = self.loss(y, y_pred)
        # Backpropagation
        grads = g.gradient(loss, self.trainable_variables)
        # Update model parameters
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Calculate average for loss and metrics
        self.train_loss(loss)
        if self._compile_metrics is not None:
            metrics = self._compile_metrics(y, y_pred)
            self.train_metrics(metrics)

    def _val_step(self, x, y):
        # Forward propagation
        y_pred = self(x)
        loss = self.loss(y, y_pred)
        # Calculate average for loss and metrics
        self.val_loss(loss)
        if self._compile_metrics is not None:
            metrics = self._compile_metrics(y, y_pred)
            self.val_metrics(metrics)

    def _test_step(self, x, y):
        # Forward propagation
        y_pred = self(x)
        loss = self.loss(y, y_pred)
        # Calculate average for loss and metrics
        self.test_loss(loss)
        if self._compile_metrics is not None:
            metrics = self._compile_metrics(y, y_pred)
            self.test_metrics(metrics)

    @staticmethod
    def _split_train_validation(x, y, validation_split=0.0, validation_data=None):
        assert 1.0 > validation_split >= 0.0, "validation_split must be in range [0, 1)"

        if validation_data is None and 1.0 > validation_split > 0.0:
            indices = np.array(range(x.shape[0]))
            np.random.shuffle(indices)
            val_x, val_y = x[indices[:int(validation_split * 100)]], y[indices[:int(validation_split * 100)]]
            x, y = x[indices[int(validation_split * 100):]], y[indices[int(validation_split):]]
        elif validation_data is not None:
            val_x, val_y = validation_data[0], validation_data[1]
        else:
            val_x, val_y = None, None
        return x, y, val_x, val_y

    def _extension_fit(self,
                       x=None, y=None,
                       batch_size=None,
                       epochs=1,
                       verbose=100,
                       validation_split=0.0,
                       validation_data=None,
                       shuffle=True,
                       initial_epoch=0):
        # Guard verbose when it equal to 0
        if verbose == 0:
            verbose = 1e+10

        # Manage train/validation set
        x, y, val_x, val_y = self._split_train_validation(x, y, validation_split, validation_data)

        # Create Dataset for training set
        training_set = tf.data.Dataset.from_tensor_slices((x, y))
        training_set = training_set.cache()
        if shuffle:
            training_set = training_set.shuffle(x.shape[0])
        if batch_size is not None:
            training_set = training_set.batch(batch_size)
        training_set = training_set.prefetch(tf.data.experimental.AUTOTUNE)

        # Create Dataset for validation set
        if val_x is not None or val_y is not None:
            validation_set = tf.data.Dataset.from_tensor_slices((val_x, val_y))
            if batch_size is not None:
                validation_set = validation_set.batch(batch_size)
        else:
            validation_set = None

        # Initial history
        history = {"loss": [], "metrics": [], "val_loss": [], "val_metrics": []}

        # Start training
        for epoch in range(initial_epoch, initial_epoch + epochs):
            # Reset states
            self.train_loss.reset_states()
            self.train_metrics.reset_states()
            self.val_loss.reset_states()
            self.val_metrics.reset_states()
            # Start timing
            start_time = time.time()
            # Train model
            for batch, (x, y) in enumerate(training_set):
                self._train_step(x, y)
                # Record to history
                history["loss"].append(self.train_loss.result())
                if self.train_metrics is not None:
                    history["metrics"].append(self.train_metrics.result())
                # Report
                if (batch + 1) % verbose == 0:
                    report_string = "Epoch {} Batch {}: Loss {:.4f}".format(epoch + 1, batch + 1, self.train_loss.result())
                    if self.train_metrics is not None:
                        report_string += ", Metrics {:.4f}".format(self.train_metrics.result())
                    report_string += ", Time {:.4f}".format(time.time() - start_time)
                    print(report_string)
            # Validation model
            if validation_set is not None:
                for batch, (x, y) in enumerate(validation_set):
                    self._val_step(x, y)
                # Record to history
                history["val_loss"].append(self.val_loss.result())
                if self.val_metrics is not None:
                    history["val_metrics"].append(self.val_metrics.result())
                # Report
                report_string = "Epoch {} Validation loss {:.4f}".format(epoch + 1, self.val_loss.result())
                if self.val_metrics is not None:
                    report_string += ", Validation metrics {:.4f}".format(self.val_metrics.result())
                print(report_string)
        return history

    def fit(self, extension_model=True, **kwargs):
        """ If extension_model is True: Use custom fit function """
        assert self.is_compiled, "Please compile model before call this function"
        if extension_model:
            return self._extension_fit(**kwargs)
        else:
            return super().fit(**kwargs)

    def _extension_evaluate(self,
                            x=None,
                            y=None,
                            batch_size=None):
        # Create Dataset
        test_set = tf.data.Dataset.from_tensor_slices((x, y))
        test_set = test_set.cache()
        if batch_size is not None:
            test_set = test_set.batch(batch_size)
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

        # Reset states
        self.test_loss.reset_states()
        self.test_metrics.reset_states()

        # Evaluate
        for batch, (x, y) in test_set:
            self._test_step(x, y)

        # Return result
        return self.test_loss.result(), self.test_metrics.result()

    def evaluate(self, extension_model=True, **kwargs):
        assert self.is_compiled, "Please compile model before call this function"
        if extension_model:
            return self._extension_evaluate(**kwargs)
        else:
            return super().evaluate(**kwargs)


