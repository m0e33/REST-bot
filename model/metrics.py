"""Collection of metrics for model evaluation"""
import logging
import tensorflow as tf


class Metrics:

  def __init__(self):
    self.train_loss_results = []
    self.train_accuracy_results = []
    self.train_mae_results = []
    self.train_rmse_results = []

    self.val_loss_results = []
    self.val_accuracy_results = []
    self.val_mae_results = []
    self.val_rmse_results = []

  def reset(self):
    self.train_epoch_mae = tf.keras.metrics.Mean()
    self.train_epoch_rmse = tf.keras.metrics.RootMeanSquaredError()
    self.train_epoch_loss_avg = tf.keras.metrics.MeanAbsoluteError()
    self.train_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    self.val_epoch_mae = tf.keras.metrics.Mean()
    self.val_epoch_rmse = tf.keras.metrics.RootMeanSquaredError()
    self.val_epoch_loss_avg = tf.keras.metrics.MeanAbsoluteError()
    self.val_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  def update_train_metrics(self, loss_value, y_batch_train, y_predict):
    self.train_epoch_loss_avg.update_state(loss_value, y_predict)
    self.train_epoch_accuracy.update_state(y_batch_train, y_predict)
    self.train_epoch_mae.update_state(y_batch_train, y_predict)
    self.train_epoch_rmse.update_state(y_batch_train, y_predict)

  def update_val_metric(self, loss_value, y_batch_val, y_predict):
    self.val_epoch_loss_avg.update_state(loss_value, y_predict)
    self.val_epoch_mae.update_state(y_batch_val, y_predict)
    self.val_epoch_rmse.update_state(y_batch_val, y_predict)
    self.val_epoch_accuracy.update_state(y_batch_val, y_predict)

  def log_final_state(self):
    self.val_loss_results.append(self.val_epoch_loss_avg.result())
    self.val_accuracy_results.append(self.val_epoch_accuracy.result())
    self.val_mae_results.append(self.val_epoch_mae.result())
    self.val_rmse_results.append(self.val_epoch_rmse.result())

    self.train_loss_results.append(self.train_epoch_loss_avg.result())
    self.train_accuracy_results.append(self.train_epoch_accuracy.result())
    self.train_mae_results.append(self.train_epoch_mae.result())
    self.train_rmse_results.append(self.train_epoch_rmse.result())

  def print_epoch_state(self, epoch):
    logging.info("Epoch {:01d} done: TRAIN: Loss: {:.3f}, Accuracy: {:.1%}, MAE: {:.3f}, RMSE: {:.3f} \n"
                 "----------------------VALIDATION: Loss: {:.3f}, Accuracy: {:.1%}, MAE: {:.3f}, RMSE: {:.3f}".format(epoch,
                                                                                                              self.train_epoch_loss_avg.result(),
                                                                                                              self.train_epoch_accuracy.result(),
                                                                                                              self.train_epoch_mae.result(),
                                                                                                              self.train_epoch_rmse.result(),
                                                                                                              self.val_epoch_loss_avg.result(),
                                                                                                              self.val_epoch_accuracy.result(),
                                                                                                              self.val_epoch_mae.result(),
                                                                                                              self.val_epoch_rmse.result()))

  def get_dictionary(self):
    return {
      "train_loss": self.train_loss_results,
      "train_mean_absolute_error": self.train_mae_results,
      "train_root_mean_squared_error": self.train_rmse_results,
      "train_accuracy": self.train_accuracy_results,
      "val_loss": self.val_loss_results,
      "val_mean_absolute_error": self.val_mae_results,
      "val_root_mean_squared_error": self.val_rmse_results,
      "val_accuracy": self.val_accuracy_results,
    }
