"""
We have essentially 3 types of data: Press release data, historical
price data and relation data. The specifics of those data
types are handled here.
"""
from typing import List
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from data.api_adapter import APIAdapter
from data.utils import build_holder_relation
from data.data_configuration import DataConfiguration

class BaseDataInfo(ABC):
    """Base class for basic data, e.g price, press, stock news"""

    def __init__(self, base_path: str, api: APIAdapter):
        self._base_path = base_path
        self._api = api
        self._path = ""

    def get_path(self, symbol):
        """Returns file path for data"""
        return f"{self._path}{symbol}.csv"

    @abstractmethod
    def get_data(self, symbol):
        """Get's data for a symbol"""


class PriceDataInfo(BaseDataInfo):
    """
    This is the ground truth for our prediction.
    """

    def __init__(self, base_path, api: APIAdapter, data_cfg: DataConfiguration):
        super().__init__(base_path, api)
        self.start = datetime.strftime(data_cfg.start,
                                       data_cfg.DATE_FORMAT)
        self.end = datetime.strftime(data_cfg.end + timedelta(days=1),
                                     data_cfg.DATE_FORMAT)
        self._path = f"{self._base_path}prices_{self.start}_{self.end}_"

    fields = ["date", "open", "close", "high", "low", "vwap"]

    def get_data(self, symbol: str):
        """Get price data data via api"""
        return self._api.get_historical_prices(symbol, self.start, self.end)[
            "historical"
        ]


class PressDataInfo(BaseDataInfo):
    """Information for Press Release Data"""

    def __init__(self, base_path, api: APIAdapter):
        super().__init__(base_path, api)
        self._path = f"{self._base_path}press_limit={self.limit}_"

    limit = 20000
    fields = ["symbol", "date", "title", "text"]

    def get_data(self, symbol: str):
        """Get press release data via api"""
        return self._api.get_press_releases(symbol, self.limit)


class StockNewsDataInfo(BaseDataInfo):
    """Information for Stock News Data"""

    def __init__(self, base_path, api: APIAdapter):
        super().__init__(base_path, api)
        self._path = f"{self._base_path}stock_news_limit={self.limit}_"

    limit = 20000
    fields = ["symbol", "publishedDate", "title", "text", "site", "url"]

    def get_data(self, symbol: str):
        """Get press release data via api"""
        return self._api.get_stock_news(symbol, self.limit)


class BaseRelationDataInfo(ABC):
    """Base class for relational data, e.g industry, stock peers, holders"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        self._base_path = base_path
        self._api = api
        self.symbols = symbols
        self._path = ""
        self.fields = ["symbol"] + symbols

    def get_path(self):
        """Returns file path for data"""
        return self._path

    @abstractmethod
    def get_data(self):
        """Get's data for all symbols and build relation matrix"""


class IndustryRelationDataInfo(BaseRelationDataInfo):
    """Information for representing industry relation between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        super().__init__(base_path, api, symbols)
        self._path = f"{self._base_path}relation_industry.csv"

    def get_data(self):
        """Get industry relation of symbols via api"""

        companies = filter(
            None,
            [self._api.get_industry_classification(symbol) for symbol in self.fields],
        )
        symbols_industries = {
            company[0]["symbol"]: company[0]["industryTitle"] for company in companies
        }

        industry_data = []
        for symbol in self.symbols:
            industry_dict = {}
            industry_dict["symbol"] = symbol
            for company, industry in symbols_industries.items():
                industry_dict[company] = (
                    1 if symbols_industries[symbol] == industry else 0
                )
            industry_data.append(industry_dict)

        return industry_data


class StockPeerRelationDataInfo(BaseRelationDataInfo):
    """Information to represent stock peer relations between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        super().__init__(base_path, api, symbols)
        self._path = f"{self._base_path}relation_peers.csv"

    def get_data(self):
        """Get stock peer relation of symbols via api"""

        companies = list(
            filter(None, [self._api.get_stock_peers(symbol) for symbol in self.fields])
        )
        company_peers = {
            company[0]["symbol"]: company[0]["peersList"] for company in companies
        }

        peer_data = []
        for symbol in self.symbols:
            peer_dict = {}
            peer_dict["symbol"] = symbol
            for company, peers in company_peers.items():
                peer_dict[company] = 1 if symbol in peers else 0
            peer_data.append(peer_dict)

        return peer_data


class InstitutionalHoldersRelationDataInfo(BaseRelationDataInfo):
    """Information to represent institutional holder relations between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        super().__init__(base_path, api, symbols)
        self._path = f"{self._base_path}relation_instholders.csv"

    def get_data(self):
        """Get institutional holders relation of symbols via api"""

        return build_holder_relation(
            self.symbols, self._api.get_institutional_holders, self.fields, threshold=2
        )


class MutualHoldersRelationDataInfo(BaseRelationDataInfo):
    """Information to represent mutual holder relations between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        super().__init__(base_path, api, symbols)
        self._path = f"{self._base_path}relation_mutualholders.csv"

    def get_data(self):
        """Get mutual holders relation of symbols via api"""

        return build_holder_relation(
            self.symbols, self._api.get_institutional_holders, self.fields, threshold=5
        )
