import tensorflow as tf
import numpy as np


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


class F1ScoreCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to monitor F1 score during training and implement early stopping
    based on F1 score improvements, with patience. This callback saves the model's best weights
    based on the highest F1 score and restores them at the end of training.
    """
    def __init__(self, validation_data, patience=5):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.best_weights = None
        self.best_f1_score = -np.inf
        self.patience = patience
        self.epochs_since_improvement = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Calculates the F1 score on the validation set
        using the model's predictions. If the F1 score improves, it saves the model's weights.
        If the F1 score has not improved for 'patience' number of epochs, it stops training.
        """
        # Get the validation data
        val_data, val_labels = self.validation_data

        # Get the model's predictions for the validation data
        val_predictions = (self.model.predict(val_data) > 0.5).astype(int)

        # Calculate precision and recall
        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()

        precision.update_state(val_labels, val_predictions)
        recall.update_state(val_labels, val_predictions)

        precision_value = precision.result().numpy()
        recall_value = recall.result().numpy()

        # Calculate F1 score
        if precision_value + recall_value > 0:
            val_f1_score = f1_score(precision_value, recall_value)
        else:
            val_f1_score = 0.0

        print(f"Epoch {epoch + 1}: val_F1_score: {val_f1_score:.4f}")

        # Check if the F1 score has improved
        if val_f1_score > self.best_f1_score:
            print(f"New best F1 score: {val_f1_score:.4f}, saving model weights...")
            self.best_f1_score = val_f1_score
            self.best_weights = self.model.get_weights()
            self.epochs_since_improvement = 0  # Reset patience counter
        else:
            self.epochs_since_improvement += 1

        # Check if we've exceeded patience
        if self.epochs_since_improvement >= self.patience:
            print(
                f"Early stopping after {epoch + 1} epochs due to no improvement in F1 score for {self.patience} epochs.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        """
        Called at the end of training. Restores the model's weights from the epoch with the best F1 score.
        """
        if self.best_weights is not None:
            print(f"Restoring model weights from epoch with the best F1 score: {self.best_f1_score:.4f}")
            self.model.set_weights(self.best_weights)
