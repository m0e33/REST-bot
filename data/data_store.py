"""DataStore for handling data delivery"""

import os
import shutil
from typing import List

from data.api_adapter import APIAdapter, get_historical_prices, get_press_release_data
from data.csv_writer import write_csv, read_csv_to_json_array

RELEVANT_HIST_FIELDS = ["date", "open", "close", "high", "low", "vwap"]
RELEVANT_PRESS_FIELDS = ["symbol", "date", "title", "text"]


def flush_store_files():
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


def _get_path(symbol: str, start: str, end: str, file_type: str = "csv"):
    return f"{DataStore.STORAGE_PATH}{symbol}_{start}_{end}.{file_type}"


def _get_path(symbol: str, limit: int, file_type: str = "csv"):
    return f"{DataStore.STORAGE_PATH}{symbol}_{limit}.{file_type}"


def _check_file(name: str):
    return os.path.exists(name)


class DataStore:
    """DataStore for handling data delivery and caching API requests"""

    STORAGE_PATH = "./data/storage/"

    def __init__(self, symbols: List[str], start: str, end: str) -> None:
        self.api = APIAdapter()
        self.symbols = symbols
        self.start = start
        self.end = end
        self._limit = 100

    def rebuild(self):
        """Clears cache and fetches data from api again"""

        flush_store_files()
        self.build()

    def build(self):
        """Writes all necessary data to the filesystem, if it is not yet present"""

        for symbol in self.symbols:
            self._build_symbol_data(symbol)

    def _build_symbol_data(self, symbol: str):
        self._build_event_data(symbol)
        self._build_historical_data(symbol)

    def _build_historical_data(self, symbol: str):
        path = _get_path(symbol, self.start, self.end)
        if not _check_file(path):
            prices = get_historical_prices(symbol, self.start, self.end)
            write_csv(path, prices, RELEVANT_HIST_FIELDS)

    def _build_event_data(self, symbol):
        path = _get_path(symbol, self._limit)
        if not _check_file(path):
            prices = get_press_release_data(symbol, self._limit)
            write_csv(path, prices, RELEVANT_PRESS_FIELDS)

    def get_press_release_data(self, symbol: str):
        """Get historical press release data from file or from API"""

        assert (
            symbol in self.symbols
        ), f"symbol {symbol} is not contained in data store."

        path = _get_path(symbol, self._limit)

        if _check_file(path):
            press_data = read_csv_to_json_array(path, RELEVANT_PRESS_FIELDS)
        else:
            press_releases = get_press_release_data(symbol, self._limit)
            write_csv(path, press_releases, RELEVANT_PRESS_FIELDS)

            press_data = read_csv_to_json_array(path, RELEVANT_PRESS_FIELDS)

        return press_data

    def get_price_data(self, symbol: str):
        """Get historical price data from file or from API"""

        assert (
            symbol in self.symbols
        ), f"symbol {symbol} is not contained in data store."

        path = _get_path(symbol, self.start, self.end)

        if _check_file(path):
            price_data = read_csv_to_json_array(path, RELEVANT_HIST_FIELDS)
        else:
            prices = get_historical_prices(symbol, self.start, self.end)
            write_csv(path, prices, RELEVANT_HIST_FIELDS)

            price_data = read_csv_to_json_array(path, RELEVANT_HIST_FIELDS)

        return price_data
