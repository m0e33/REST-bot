"""Executable for testing the functionality of various methods and modules"""

from tensorflow.keras import Sequential
from model.layers import TypeSpecificEncoder
from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from model.configuration import TrainConfiguration, HyperParameterConfiguration
from model.model import RESTNet


if __name__ == "__main__":
    data_cfg = DataConfiguration(
        symbols=["AAPL", "ACN", "CDW", "NFLX"],
        start="2020-12-29",
        end="2021-04-06",
        feedback_metrics=["open", "close", "high", "low", "vwap"],
        stock_context_days=3,
    )

    train_cfg = TrainConfiguration()

    hp_cfg = HyperParameterConfiguration()

    data_store = DataStore(data_cfg)
    data_store.build()

    prepro = Preprocessor(data_store, data_cfg, train_cfg)
    prepro.build_events_data_with_gt()

    train_ds = prepro.get_train_ds()
    val_ds = prepro.get_val_ds()
    test_ds = prepro.get_test_ds()

    model = RESTNet(hp_cfg)
    model.compile()
    model.run_eagerly = True

    for example_inputs, example_labels in train_ds.take(1):
        print(
            f"Inputs shape (batch, dates, symbols, events, events words, word embeddings): "
            f"{example_inputs.shape}"
        )
        print(f"Labels shape (batch, symbols, gt_trend): {example_labels.shape}")

        one_input = example_inputs[0]
        x = model.predict(one_input)
        print(x.shape)
