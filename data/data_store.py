"""DataStore for handling data delivery"""

import os
import shutil
from typing import List

from data.api_adapter import APIAdapter
from data.csv_writer import write_csv, read_csv_to_json_array
from data.data_info import PriceDataInfo, PressDataInfo

PRICE_DATA = 'price_data'
PRESS_DATA = 'press_data'


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
        self._data_info = {
            PRICE_DATA: PriceDataInfo(DataStore.STORAGE_PATH),
            PRESS_DATA: PressDataInfo(DataStore.STORAGE_PATH)
        }

    def rebuild(self):
        """Clears cache and fetches data from api again"""

        flush_store_files()
        self.build()

    def build(self):
        """Writes all necessary data to the filesystem, if it is not yet present"""

        for symbol in self.symbols:
            self._build_data_for_symbol(symbol)

    def _build_data_for_symbol(self, symbol: str):
        self._build_event_data(symbol)
        self._build_historical_price_data(symbol)

    def _build_historical_price_data(self, symbol: str):
        price_data = self._data_info[PRICE_DATA]

        path = price_data.get_path(symbol)

        if not _check_file(path):
            prices = self.api.get_historical_prices(symbol, price_data.start, price_data.end)['historical']
            write_csv(path, prices, price_data.fields)

    def _build_event_data(self, symbol):
        press_data = self._data_info[PRESS_DATA]

        path = press_data.get_path(symbol)

        if not _check_file(path):
            prices = self.api.get_press_release_data(symbol, press_data.limit)
            write_csv(path, prices, press_data.fields)

    def get_press_release_data(self, symbol: str):
        """Get historical press release data from file or from API"""
        press_data = self._data_info[PRESS_DATA]

        assert (
                symbol in self.symbols
        ), f"DataStore does not contain symbol '{symbol}'."

        path = press_data.get_path(symbol)

        if _check_file(path):
            press_data = read_csv_to_json_array(path, press_data.fields)
        else:
            press_releases = self.api.get_press_release_data(symbol, press_data.limit)
            write_csv(path, press_releases, press_data.fields)

            press_data = read_csv_to_json_array(path, press_data.fields)

        return press_data

    def get_price_data(self, symbol: str):
        """Get historical price data from file or from API"""
        price_data = self._data_info[PRICE_DATA]

        assert (
                symbol in self.symbols
        ), f"symbol {symbol} is not contained in data store."

        path = price_data.get_path(symbol)

        if _check_file(path):
            price_data = read_csv_to_json_array(path, price_data.fields)
        else:
            prices = self.api.get_historical_prices(symbol, price_data.start, price_data.end)
            write_csv(path, prices, price_data.fields)

            price_data = read_csv_to_json_array(path, price_data.fields)

        return price_data
