import datetime
import logging
import time
from typing import Union, List
import numpy as np
import tensorflow as tf
from model.model import RESTNet

from kubeflow_utils.kubeflow_serve import KubeflowServe
from kubeflow_utils.metadata_config import MetadataConfig
from kubeflow_utils.training_result import TrainingResult
from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from configuration.configuration import TrainConfiguration, HyperParameterConfiguration

from utils.progess import Progress
from utils.symbols import load_symbols
from configuration.configuration import (
    TrainConfiguration,
    HyperParameterConfiguration,
    hp_cfg_is_cached,
    deserialize_hp_cfg,
    serialize_hp_cfg,
    train_cfg_is_cached,
    deserialize_train_cfg,
    serialize_train_cfg,
)

logger = logging.getLogger("kubeflow_adapter")
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H꞉%M꞉%S")


class KubeflowAdapter(KubeflowServe):

    def __init__(self, num_gpus: int):
        super().__init__()
        self.num_gpus = num_gpus

    def download_data_component(self, cloud_path: str, data_path: str):
        """Download data component"""
        self.download_data(cloud_path, data_path)

    @staticmethod
    def create_datasets(data_cfg: DataConfiguration, train_cfg: TrainConfiguration, hp_cfg: HyperParameterConfiguration, global_batch_size):
        """Read input data and create train, validation and test datasets."""

        logger.info("Reading Symbols")

        logger.info(f"Data configuration: {str(data_cfg)}")

        data_store = DataStore(data_cfg)

        logger.info("Build Data Store")
        data_store.build()

        logger.info("Preprocessing Data")
        prepro = Preprocessor(data_store, data_cfg, train_cfg, hp_cfg)
        prepro.build_events_data_with_gt()
        logger.info("Finished Preprocessing Data")

        logger.info("Build Datasets")
        train_ds = prepro.get_train_ds(global_batch_size)
        val_ds = prepro.get_val_ds(global_batch_size)
        test_ds = prepro.get_test_ds(global_batch_size)
        logger.info("Finished bulding Datasets")

        logger.info("Data is ready")
        return train_ds, val_ds, test_ds

    def get_metadata(self) -> MetadataConfig:
        return MetadataConfig(
            model_names="RESTNet",
            model_description="custom model for predicting stock prices",
            model_version="v0.1",
            model_type="linear_regression",
            maturity_state="development",
            dataset_name="rest_data",
            dataset_description="press releases, news and historical prices",
            dataset_path="rest_data",
            dataset_version="v0.1",
            owner="developer@mail.me",
            training_framework_name="tensorflow",
            training_framework_version="1.2.1",
        )

    def train_model(self, pipeline_run=False, data_path: str = "") -> TrainingResult:

        train_cfg = TrainConfiguration()

        strategy = tf.distribute.MirroredStrategy()
        logger.info(f'Number of devices: {strategy.num_replicas_in_sync}')
        global_batch_size = train_cfg.batch_size * strategy.num_replicas_in_sync
        logger.info(f"Global Batch Size: {global_batch_size}")

        hp_cfg = HyperParameterConfiguration()
        data_cfg = DataConfiguration(
            symbols=load_symbols(4),
            start="2021-01-01",
            end="2021-08-01",
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            stock_news_fetch_limit=200,
            events_per_day_limit=10
        )
        logger.info(f"Train configuration: {str(train_cfg)}")
        logger.info(f"Hyperparameter configuration: {str(hp_cfg)}")
        logger.info(f"Data configuration: {str(data_cfg)}")

        # Define summary writers
        train_log_dir = f'logs/gradient_tape/{current_time}/train'
        val_log_dir = f'logs/gradient_tape/{current_time}/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(val_log_dir)

        train_ds, val_ds, test_ds = self.create_datasets(data_cfg, train_cfg, hp_cfg, global_batch_size)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
        val_dist_dataset = strategy.experimental_distribute_dataset(val_ds)
        test_dist_dataset = strategy.experimental_distribute_dataset(test_ds)

        logger.info("Starting training..")
        progress = Progress(hp_cfg.num_epochs)

        with strategy.scope():
            model = RESTNet(hp_cfg, train_cfg)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def compute_loss(labels, predictions):
                per_example_loss = loss_object(labels, predictions)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

            train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
            val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
            test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

            def train_step(inputs):
                x, y = inputs
                with tf.GradientTape() as tape:
                    predictions = model(x, training=True)
                    loss = compute_loss(y, predictions)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                train_loss.update_state(loss)
                return loss

            def val_step(inputs):
                x, y = inputs
                predictions = model(x)
                loss = loss_object(y, predictions)
                val_loss.update_state(loss)

            def test_step(inputs):
                x, y = inputs
                predictions = model(x)
                loss = loss_object(y, predictions)
                test_loss.update_state(loss)

        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_loss = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

        @tf.function
        def distributed_val_step(dataset_inputs):
            return strategy.run(val_step, args=(dataset_inputs,))

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))

        for epoch in range(hp_cfg.num_epochs):
            logger.info(f"Started Epoch {epoch+1} from {hp_cfg.num_epochs}")
            start_time = time.time()

            if epoch == 10:
                tf.profiler.experimental.start(f"logs/profiler/{current_time}")

            if epoch % 100 == 0 | epoch == 1:
                path = f"{self.get_model_path()}-{epoch:04d}"
                logger.info(f"saving model to '{path}'")
                model.save_weights(path)

            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0
            for inputs in train_dist_dataset:
                total_loss += distributed_train_step(inputs)
                num_batches += 1
            train_loss = total_loss / num_batches

            # VALIDATION LOOP
            for inputs in val_dist_dataset:
                distributed_val_step(inputs)

            if epoch == 20:
                tf.profiler.experimental.stop()

            step_duration = time.time() - start_time
            progress.step(step_duration)
            logger.info(progress.eta(epoch))

            with train_summary_writer.as_default():
                tf.summary.scalar('step_duration', step_duration, step=epoch)
                tf.summary.scalar('train_loss', train_loss, step=epoch)
                tf.summary.scalar('val_loss', val_loss.result(), step=epoch)

            logger.info(f"Epoch {epoch}, loss: {train_loss}, val_loss: {val_loss.result()}")
            val_loss.reset_states()

        # TEST LOOP
        for inputs in test_dist_dataset:
            distributed_test_step(inputs)

        # save final model
        return TrainingResult(
            models=[model],
            evaluation={'test_loss': test_loss.result(), 'val_loss': val_loss.result(), 'train_loss': train_loss},
            hyperparameters={}
        )

    def predict_model(self, model: RESTNet, data) -> any:
        """Predict using the model for given ndarray."""
        prediction = model.predict(data=data)

        return prediction

    def load_model(self):
        # This works as long as train and hyper parameter config are the default values.
        if hp_cfg_is_cached():
            hp_cfg = deserialize_hp_cfg()
        else:
            hp_cfg = HyperParameterConfiguration()
        if train_cfg_is_cached():
            train_cfg = deserialize_train_cfg()
        else:
            train_cfg = TrainConfiguration()
        model = RESTNet(hp_cfg, train_cfg)

        model.load_weights(self.get_model_path())

        return model
