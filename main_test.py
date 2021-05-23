"""Executable for testing the functionality of various methods and modules"""

import data.data_store

if __name__ == "__main__":
    data.data_store.flush()
    data.data_store.get_price_data("AAPL", "2021-01-01", "2021-05-22")
