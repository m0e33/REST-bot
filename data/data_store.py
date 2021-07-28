"""DataStore for handling data delivery"""

import os
import shutil
from enum import Enum

from data.api_adapter import APIAdapter
from data.csv_writer import write_csv, read_csv_to_json_array
from data.data_info import (
    PriceDataInfo,
    PressDataInfo,
    IndustryRelationDataInfo,
    StockPeerRelationDataInfo,
    InstitutionalHoldersRelationDataInfo,
    StockNewsDataInfo,
    MutualHoldersRelationDataInfo,
)

from configuration.data_configuration import DataConfiguration, serialize_data_cfg, deserialize_data_cfg, data_cfg_is_cached


class DataType(Enum):
    """To distinguish between different types of data"""

    PRICE_DATA = "price_data"
    PRESS_DATA = "press_data"
    INDUSTRY_RELATION_DATA = "industry_relation_data"
    STOCK_PEER_RELATION_DATA = "stock_peer_relation_data"
    INSTITUTIONAL_HOLDERS_RELATION_DATA = "inst_holders_relation_data"
    STOCK_NEWS_DATA = "stock_news_data"
    MUTUAL_HOLDERS_RELATION_DATA = "mutual_holders_relation_data"


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

    def __init__(self, data_cfg: DataConfiguration) -> None:
        self.api = APIAdapter()
        self.data_cfg = data_cfg

        self.old_data_can_be_reused = self._data_config_has_not_been_changed()
        print("Data store reusable (from what we can tell from configs): " + str(self.old_data_can_be_reused))
        serialize_data_cfg(self.data_cfg)

        self._basic_data_info = {
            DataType.PRICE_DATA: PriceDataInfo(DataStore.STORAGE_PATH, self.api, self.data_cfg),
            DataType.PRESS_DATA: PressDataInfo(DataStore.STORAGE_PATH, self.api, self.data_cfg.stock_news_limit),
            DataType.STOCK_NEWS_DATA: StockNewsDataInfo(
                DataStore.STORAGE_PATH, self.api, self.data_cfg.stock_news_limit
            ),
        }
        self._relation_data_info = {
            DataType.INDUSTRY_RELATION_DATA: IndustryRelationDataInfo(
                DataStore.STORAGE_PATH, self.api, self.data_cfg.symbols
            ),
            DataType.STOCK_PEER_RELATION_DATA: StockPeerRelationDataInfo(
                DataStore.STORAGE_PATH, self.api, self.data_cfg.symbols
            ),
            DataType.INSTITUTIONAL_HOLDERS_RELATION_DATA: InstitutionalHoldersRelationDataInfo(
                DataStore.STORAGE_PATH, self.api, self.data_cfg.symbols
            ),
            DataType.MUTUAL_HOLDERS_RELATION_DATA: MutualHoldersRelationDataInfo(
                DataStore.STORAGE_PATH, self.api, self.data_cfg.symbols
            ),
        }

    def rebuild(self):
        """Clears cache and fetches data from api again"""

        flush_store_files()
        self.build()

    def build(self):
        """Writes all necessary data to the filesystem, if it is not yet present"""

        # Get price and press data for each symbols
        for symbol in self.data_cfg.symbols:
            self._maybe_build_data_for_symbol(symbol, DataType.PRESS_DATA)
            self._maybe_build_data_for_symbol(symbol, DataType.PRICE_DATA)
            self._maybe_build_data_for_symbol(symbol, DataType.STOCK_NEWS_DATA)

        # Get relation data for all symbols
        self._maybe_build_data_for_symbols(DataType.INDUSTRY_RELATION_DATA)
        self._maybe_build_data_for_symbols(DataType.STOCK_PEER_RELATION_DATA)
        self._maybe_build_data_for_symbols(DataType.INSTITUTIONAL_HOLDERS_RELATION_DATA)
        self._maybe_build_data_for_symbols(DataType.MUTUAL_HOLDERS_RELATION_DATA)

    def get_price_data(self, symbol: str):
        """Get historical price data from file or from API"""
        return self._get_basic_data_for_symbol_from_file_or_rebuild(
            symbol, DataType.PRICE_DATA
        )

    def get_press_release_data(self, symbol: str):
        """Get historical press release data from file or from API"""
        return self._get_basic_data_for_symbol_from_file_or_rebuild(
            symbol, DataType.PRESS_DATA
        )

    def get_industry_relation_data(self):
        """Get industry relation data from file or from API"""
        return self._get_relation_data_from_file_or_rebuild(
            DataType.INDUSTRY_RELATION_DATA
        )

    def get_stock_peer_relation_data(self):
        """Get stock peer relation data from file or from API"""
        return self._get_relation_data_from_file_or_rebuild(
            DataType.STOCK_PEER_RELATION_DATA
        )

    def get_institutional_holder_relation_data(self):
        """Get institutional holders relation data from file or from API"""
        return self._get_relation_data_from_file_or_rebuild(
            DataType.INSTITUTIONAL_HOLDERS_RELATION_DATA
        )

    def get_stock_news_data(self, symbol: str):
        """Get stock news data for all symbols"""
        return self._get_basic_data_for_symbol_from_file_or_rebuild(
            symbol, DataType.STOCK_NEWS_DATA
        )

    def get_mutual_holder_relation_data(self):
        """Get mutual holders relation data from file or from API"""
        return self._get_relation_data_from_file_or_rebuild(
            DataType.MUTUAL_HOLDERS_RELATION_DATA
        )

    def _maybe_build_data_for_symbol(self, symbol: str, data_type: DataType):
        data_info = self._basic_data_info[data_type]

        path = data_info.get_path(symbol)
        if not _check_file(path) or not self.old_data_can_be_reused:
            self.old_data_can_be_reused = False
            data = data_info.get_data(symbol)
            write_csv(path, data, data_info.fields)

    def _maybe_build_data_for_symbols(self, data_type: DataType):
        data_info = self._relation_data_info[data_type]

        path = data_info.get_path()
        if not _check_file(path) or not self.old_data_can_be_reused:
            self.old_data_can_be_reused = False
            data = data_info.get_data()
            write_csv(path, data, data_info.fields)

    def _get_basic_data_for_symbol_from_file_or_rebuild(
        self, symbol: str, data_type: DataType
    ):
        """Get historical press release data from file or from API"""
        data_info = self._basic_data_info[data_type]
        assert symbol in self.data_cfg.symbols, f"DataStore does not contain symbol '{symbol}'."

        self._maybe_build_data_for_symbol(symbol, DataType.PRESS_DATA)
        data_info = read_csv_to_json_array(data_info.get_path(symbol), data_info.fields)

        return data_info

    def _get_relation_data_from_file_or_rebuild(self, data_type: DataType):
        """Get historical press release data from file or from API"""
        data_info = self._relation_data_info[data_type]

        self._maybe_build_data_for_symbols(DataType.PRESS_DATA)
        data = read_csv_to_json_array(data_info.get_path(), data_info.fields)

        return data

    def _data_config_has_not_been_changed(self):
        if data_cfg_is_cached():
            old_cfg = deserialize_data_cfg()
            return old_cfg == self.data_cfg
        return False
