"""Executable for testing the functionality of various methods and modules"""

from data.data_store import DataStore
from data.preprocesser import Preprocessor

if __name__ == "__main__":
    symbols = ["AAPL", "ACN", "CDW", "NFLX"]
    data_store = DataStore(symbols, "2021-01-01", "2021-04-01")
    data_store.build()

    prepro = Preprocessor(data_store)
    prepro.build_event_data()
