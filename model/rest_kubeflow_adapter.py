import datetime
import logging
import time
from typing import Union, List
import numpy as np
import tensorflow as tf
from model.model import RESTNet
import os
import yaml
from dataclasses import asdict

from kubeflow_utils.kubeflow_serve import KubeflowServe
from kubeflow_utils.metadata_config import MetadataConfig
from kubeflow_utils.training_result import TrainingResult
from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from configuration.configuration import TrainConfiguration, HyperParameterConfiguration
from model.metrics import Metrics

# tf.config.run_functions_eagerly(False)
from utils import load_symbols
from utils.progess import Progress

logger = logging.getLogger("kubeflow_adapter")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H;%M;%S")


class KubeflowAdapter(KubeflowServe):

    def __init__(self):
        super().__init__()

    def download_data_component(self, cloud_path: str, data_path: str):
        """Download data component"""
        self.download_data(cloud_path, data_path)

    def read_input(self, train_cfg: TrainConfiguration, hp_cfg: HyperParameterConfiguration):
        """Read input data and split it into train and test."""

        logger.info("Reading Symbols")

        data_cfg = DataConfiguration(
            symbols=load_symbols(4),
            start="2021-01-01",
            end="2021-08-01",
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            stock_news_limit=200
        )

        logger.info(f"Data configuration: {str(data_cfg)}")

        data_store = DataStore(data_cfg)

        logger.info("Build Data Store")
        data_store.build()

        logger.info("Preprocessing Data")
        prepro = Preprocessor(data_store, data_cfg, train_cfg, hp_cfg)
        prepro.build_events_data_with_gt()
        logger.info("Finished Preprocessing Data")

        logger.info("Build Datasets")
        train_ds = prepro.get_train_ds()
        val_ds = prepro.get_val_ds()
        test_ds = prepro.get_test_ds()
        logger.info("Finished bulding Datasets")

        logger.info("Data is ready")
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
        train_cfg = TrainConfiguration()
        hp_cfg = HyperParameterConfiguration()

        logger.info(f"Train configuration: {str(train_cfg)}")
        logger.info(f"Hyperparameter configuration: {str(hp_cfg)}")

        train_ds, val_ds, test_ds = self.read_input(train_cfg, hp_cfg)

        model = RESTNet(hp_cfg, train_cfg)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        loss_object = tf.keras.losses.MeanSquaredError()

        # Define our metrics
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        def train_step(model, optimizer, x_train, y_train):
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = loss_object(y_train, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss(loss)

        def test_step(model, x_test, y_test):
            predictions = model(x_test)
            loss = loss_object(y_test, predictions)

            test_loss(loss)

        train_log_dir = f'logs/gradient_tape/{current_time}/train'
        test_log_dir = f'logs/gradient_tape/{current_time}/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        metrics = Metrics()

        logger.info("Starting training..")
        progress = Progress(hp_cfg.num_epochs)

        train_ds_iter = iter(train_ds)
        for epoch in range(hp_cfg.num_epochs):
            if epoch == 10:
                tf.profiler.experimental.start(f"logs/profiler/{current_time}")

            start_time = time.time()
            logger.info(f"Started Epoch {epoch+1} from {hp_cfg.num_epochs}")

            metrics.reset()

            # Training loop
            # with tf.profiler.experimental.Trace("Train", step_num=step):
            for x_batch_train, y_batch_train in train_ds:
                train_step(model, optimizer, x_batch_train, y_batch_train)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

            for x_batch_val, y_batch_val in val_ds:
                test_step(model, x_batch_val, y_batch_val)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)

            #metrics.log_final_state()
            #metrics.print_epoch_state(epoch, logger)

            step_duration = time.time() - start_time
            with train_summary_writer.as_default():
                tf.summary.scalar('step_duration', step_duration, step=epoch)

            if epoch == 20:
                tf.profiler.experimental.stop()

            progress.step(step_duration)
            logger.info(progress.eta(epoch))


        return TrainingResult(
            models=[],
            evaluation=metrics.get_dictionary(),
            hyperparameters={}
        )

    def predict_model(self, models, data) -> Union[np.ndarray, List, str, bytes]:
        """Predict using the model for given ndarray."""
        prediction = models[0].predict(data=data)

        return prediction
