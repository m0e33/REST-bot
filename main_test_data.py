"""Executable for testing the functionality of various methods and modules"""

from data.data_store import DataStore, DataConfiguration
from data.preprocesser import Preprocessor

if __name__ == "__main__":
    data_configuration = DataConfiguration(
        ["AAPL", "ACN", "CDW", "NFLX"],
        "2020-12-27",
        "2021-04-06"
    )

    data_store = DataStore(data_configuration)
    data_store.build()

    prepro = Preprocessor(data_store, data_configuration)
    prepro.build_event_data()
