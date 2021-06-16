"""Executable for testing the functionality of various methods and modules"""

from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor, TrainConfiguration


if __name__ == "__main__":
    data_cfg = DataConfiguration(
        ["AAPL", "ACN", "CDW", "NFLX"],
        "2020-12-29",
        "2021-04-06",
        stock_context_days=30,
    )

    train_cfg = TrainConfiguration(val_split=0.2, test_split=0.1)

    data_store = DataStore(data_cfg)
    data_store.build()

    prepro = Preprocessor(data_store, data_cfg, train_cfg)
