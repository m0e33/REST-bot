"""Executable for testing the functionality of various methods and modules"""

from tensorflow import keras
from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from model.configuration import TrainConfiguration, HyperParameterConfiguration
from model.model import RESTNet


if __name__ == "__main__":
    data_cfg = DataConfiguration(
        symbols=["AAPL", "ACN", "CDW", "NFLX"],
        start="2020-02-03",
        end="2021-04-06",
        feedback_metrics=["open", "close", "high", "low", "vwap"],
        stock_context_days=3,
    )

    train_cfg = TrainConfiguration()

    hp_cfg = HyperParameterConfiguration()

    data_store = DataStore(data_cfg)
    data_store.rebuild()

    prepro = Preprocessor(data_store, data_cfg, train_cfg)
    prepro.build_events_data_with_gt()

    train_ds = prepro.get_train_ds()
    val_ds = prepro.get_val_ds()
    test_ds = prepro.get_test_ds()


    model = RESTNet(hp_cfg)
    # model.run_eagerly = True

    model.compile(
        optimizer=keras.optimizers.Adadelta(learning_rate=0.01),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(),
        # List of metrics to monitor
        metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.RootMeanSquaredError()],
    )

    for example_inputs, example_labels in train_ds.take(1):
        print(
            f"Inputs shape (batch, dates, symbols, events, events words, word embeddings): "
            f"{example_inputs.shape}"
        )
        print(f"Labels shape (batch, symbols, gt_trend): {example_labels.shape}")

    history = model.fit(
        train_ds,
        epochs=1,
        validation_data=val_ds
    )
