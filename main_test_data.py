"""Executable for testing the functionality of various methods and modules"""

from data.data_store import DataStore


if __name__ == "__main__":
    symbols = ["AAPL", "ACN", "CDW", "NFLX"]
    data_store = DataStore(symbols, "2021-01-01", "2021-04-01")
    data_store.rebuild()
    print(data_store.get_price_data("AAPL"))
