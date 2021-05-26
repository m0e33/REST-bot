"""DataStore for handling data delivery"""

import os
import shutil
from enum import Enum
from typing import List

from data.api_adapter import APIAdapter
from data.csv_writer import write_csv, read_csv_to_json_array
from data.data_info import PriceDataInfo, PressDataInfo


class DataType(Enum):
    """To distinguish between different types of data"""

    PRICE_DATA = "price_data"
    PRESS_DATA = "press_data"


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
            DataType.PRICE_DATA: PriceDataInfo(DataStore.STORAGE_PATH, self.api),
            DataType.PRESS_DATA: PressDataInfo(DataStore.STORAGE_PATH, self.api),
        }

    def rebuild(self):
        """Clears cache and fetches data from api again"""

        flush_store_files()
        self.build()

    def build(self):
        """Writes all necessary data to the filesystem, if it is not yet present"""
        _ = [
            [self._build_data_for_symbol(symbol, type) for type in DataType]
            for symbol in self.symbols
        ]

    def get_price_data(self, symbol: str):
        """Get historical price data from file or from API"""
        return self._get_data_from_file_or_rebuild(symbol, DataType.PRICE_DATA)

    def get_press_release_data(self, symbol: str):
        """Get historical press release data from file or from API"""
        return self._get_data_from_file_or_rebuild(symbol, DataType.PRESS_DATA)

    def _build_data_for_symbol(self, symbol: str, data_type: DataType):
        data_info = self._data_info[data_type]

        path = data_info.get_path(symbol)

        if not _check_file(path):
            data = data_info.get_data(symbol)
            write_csv(path, data, data_info.fields)

    def _get_data_from_file_or_rebuild(self, symbol: str, data_type: DataType):
        """Get historical press release data from file or from API"""
        data_info = self._data_info[data_type]

        assert symbol in self.symbols, f"DataStore does not contain symbol '{symbol}'."

        path = data_info.get_path(symbol)

        if _check_file(path):
            data_info = read_csv_to_json_array(path, data_info.fields)
        else:
            self._build_data_for_symbol(symbol, DataType.PRESS_DATA)
            data_info = read_csv_to_json_array(path, data_info.fields)

        return data_info
