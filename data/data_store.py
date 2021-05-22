"""DataStore for handling data delivery"""

import os
import shutil
from .api_adapter import APIAdapter, get_historical_prices
from .csv_writer import write

RELEVANT_HIST_FIELDS = ["date", "open", "close", "high", "low", "vwap"]


def flush():
    """Wipe all existing data files"""

    for filename in os.listdir(DataStore.STORAGE_PATH):
        file_path = os.path.join(DataStore.STORAGE_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except StopIteration as error:
            print("Failed to delete %s. Reason: %s" % (file_path, error))


def _check_file(name: str):
    return os.path.exists(name)


def _get_path(symbol: str, start: str, end: str, file_type: str = "csv"):
    return f"{DataStore.STORAGE_PATH}{symbol}_{start}_{end}.{file_type}"


def get_price_data(symbol: str, start: str, end: str):
    """Get historical price data from file or from API"""

    path = _get_path(symbol, start, end)
    if _check_file(path):
        # load file
        pass
    else:
        prices = get_historical_prices(symbol, start, end)
        write(path, prices, RELEVANT_HIST_FIELDS)


class DataStore:
    """DataStore for handling data delivery"""

    STORAGE_PATH = "./data/storage/"

    def __init__(self) -> None:
        self.api = APIAdapter()

    def get_press_release_data(self):
        """Get press release data from file or from API"""

    def add_another_method_for_linter(self):
        """Another public method for linter"""
