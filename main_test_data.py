"""Executable for testing the functionality of various methods and modules"""

from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor
from configuration.configuration import TrainConfiguration, HyperParameterConfiguration

if __name__ == "__main__":
    data_cfg = DataConfiguration(
        symbols=["AAPL", "ACN", "CDW", "NFLX"],
        start="2021-02-01",
        end="2021-04-06",
        feedback_metrics=["open", "close", "high", "low", "vwap"],
        stock_news_limit=200
    )

    train_cfg = TrainConfiguration()
    hp_cfg = HyperParameterConfiguration()

    data_store = DataStore(data_cfg)
    data_store.build()

    prepro = Preprocessor(data_store, data_cfg, train_cfg, hp_cfg)
    prepro.build_events_data_with_gt()

    ds = prepro.get_train_ds()

    for example_inputs, example_labels in ds.take(1):
        print(f'Inputs shape (dates, symboles, events, events words, word embeddings): '
              f'{example_inputs.shape}')
        print(f'Labels shape (batch, symbols, gt_trend): {example_labels.shape}')
