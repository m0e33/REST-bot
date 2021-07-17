import logging
import time
from typing import Union, List
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
from model.metrics import Metrics

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
            start="2020-12-29",
            end="2021-04-06",
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            stock_context_days=3,
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

        num_epochs = 1

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
        loss_object = tf.keras.losses.MeanSquaredError()

        def loss(model, x, y, training=False):
            # training=training is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_predict= model(x, training=training)

            return loss_object(y_true=y, y_pred=y_predict), y_predict

        def grad(model, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value, y_predict = loss(model, inputs, targets)
            return loss_value, tape.gradient(loss_value, model.trainable_variables), y_predict

        metrics = Metrics()

        for example_input, example_label in train_ds.take(1):
            loss_value, y_predict = loss(model, example_input, example_label)
            logging.info("Loss test: {}".format(loss_value.numpy))

        logging.info("Starting training..")

        for epoch in range(num_epochs):
            start_time = time.time()
            logging.info(f"Started Epoch {epoch+1} from {num_epochs}")

            metrics.reset()

            # Training loop - using batches of 32
            for x_batch_train, y_batch_train in train_ds:
                # Optimize the model
                loss_value, grads, y_predict = grad(model, x_batch_train, y_batch_train)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                metrics.update_train_metrics(loss_value, y_batch_train, y_predict)

            # End epoch

            for x_batch_val, y_batch_val in val_ds:
                loss_value, val_predict = loss(model, x_batch_val, y_batch_val)
                # Update val metrics
                metrics.update_val_metric(loss_value, y_batch_val, val_predict)

            metrics.log_final_state()
            metrics.print_epoch_state(epoch)

            logging.info("Time taken: %.2fs" % (time.time() - start_time))

        return TrainingResult(
            models=[],
            evaluation=metrics.get_dictionary(),
            hyperparameters={}
        )

    def predict_model(self, models, data) -> Union[np.ndarray, List, str, bytes]:
        """Predict using the model for given ndarray."""
        prediction = models[0].predict(data=data)

        return prediction
