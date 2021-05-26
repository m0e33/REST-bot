"""
We have essentially 3 types of data: Press release data, historical
price data and relation data. The specifics of those data
types are handled here.
"""

from dataclasses import dataclass

from data.api_adapter import APIAdapter


@dataclass
class PriceDataInfo:
    """Information for Historical Price Data"""

    def __init__(self, base_path, api: APIAdapter):
        self._base_path = base_path
        self._path = f"{self._base_path}prices_{self.start}_{self.end}_"
        self._api = api

    start = "2021-01-01"
    end = "2021-04-01"
    fields = ["date", "open", "close", "high", "low", "vwap"]

    def get_path(self, symbol):
        """File path for price data"""
        return f"{self._path}{symbol}.csv"

    def get_data(self, symbol: str):
        """Get price data data via api"""
        return self._api.get_historical_prices(symbol, self.start, self.end)[
            "historical"
        ]


@dataclass
class PressDataInfo:
    """Information for Press Release Data"""

    def __init__(self, base_path, api: APIAdapter):
        self._base_path = base_path
        self._path = f"{self._base_path}press_limit={self.limit}_"
        self._api = api

    limit = 100
    fields = ["symbol", "date", "title", "text"]

    def get_path(self, symbol):
        """File path for press release data"""
        return f"{self._path}{symbol}.csv"

    def get_data(self, symbol: str):
        """Get press release data via api"""
        return self._api.get_press_releases(symbol, self.limit)
