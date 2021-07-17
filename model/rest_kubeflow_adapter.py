import logging
from os import path
from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf
from model.model import RESTNet
import os

from kubeflow_utils.kubeflow_serve import KubeflowServe
from kubeflow_utils.metadata_config import MetadataConfig
from kubeflow_utils.training_result import TrainingResult
from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from model.configuration import TrainConfiguration, HyperParameterConfiguration

logging.basicConfig(format="%(message)s")
logging.getLogger().setLevel(logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def eval_model(model, test_X, test_y):
    """Evaluate the model performance."""
    # predictions = model.predict(test_X)
    # mean_error = mean_absolute_error(predictions, test_y)
    # logging.info("mean_absolute_error=%.2f", mean_error)
    return 0


class KubeflowAdapter(KubeflowServe):
    def __init__(self):
        super().__init__()

    def download_data_component(self, cloud_path: str, data_path: str):
        self.download_data(cloud_path, data_path)

    def read_input(self):
        """Read input data and split it into train and test."""
        data_cfg = DataConfiguration(
            symbols=["AAPL", "ACN", "CDW", "NFLX"],
            start="2019-03-29",
            end="2021-04-30",
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            stock_context_days=6,
        )

        train_cfg = TrainConfiguration()

        data_store = DataStore(data_cfg)
        data_store.build()

        print("Preprocessor -> build events data with gt")
        prepro = Preprocessor(data_store, data_cfg, train_cfg)
        prepro.build_events_data_with_gt()

        print("Preprocessor -> get datasets")
        train_ds = prepro.get_train_ds()
        val_ds = prepro.get_val_ds()
        test_ds = prepro.get_test_ds()

        print("Preprocessor -> Returning Datasets")
        return train_ds, val_ds, test_ds

    def get_metadata(self) -> MetadataConfig:
        return MetadataConfig(
            model_names="trained_ames_model.dat",
            model_description="xgboost model for predicting prices",
            model_version="v0.1",
            model_type="linear_regression",
            maturity_state="development",
            dataset_name="ames training data",
            dataset_description="dataset consists of multiple prices",
            dataset_path="ames_dataset",
            dataset_version="v0.3",
            owner="developer@mail.me",
            training_framework_name="xgboost",
            training_framework_version="1.2.1",
        )

    def train_model(self, pipeline_run=False, data_path: str = "") -> TrainingResult:
        train_ds, val_ds, test_ds = self.read_input()

        hp_cfg = HyperParameterConfiguration()

        model = RESTNet(hp_cfg)
        model.run_eagerly = True

        num_epochs = 201

        optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        def loss(model, x, y, training):
            # training=training is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_ = model(x, training=training)

            return loss_object(y_true=y, y_pred=y_)

        def grad(model, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value = loss(model, inputs, targets, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

        train_loss_results = []
        train_accuracy_results = []
        logging.info("Started training")
        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # Training loop - using batches of 32
            for x, y in train_ds:
                # Optimize the model
                loss_value, grads = grad(model, x, y)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(y, model(x, training=True))
            logging.info("epoch done")
            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 50 == 0:
                logging.info("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                            epoch_loss_avg.result(),
                                                                            epoch_accuracy.result()))

        return TrainingResult(
            models=model,
            evaluation={"mean_absolute_error": 2},
            hyperparameters={
                "hyperparam_1": 1,
                "hyperparam_2": 2,
            },
        )

    def predict_model(self, models, data) -> Union[np.ndarray, List, str, bytes]:
        """Predict using the model for given ndarray."""
        prediction = models[0].predict(data=data)

        return prediction
