"""
We have essentially 3 types of data: Press release data, historical
price data and relation data. The specifics of those data
types are handled here.
"""
from typing import List

from data.api_adapter import APIAdapter


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
        return self._api.get_press_releases(symbol[0], self.limit)


class IndustryRelationDataInfo:
    """Information to represent different relations between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        self._base_path = base_path
        self._path = f"{self._base_path}relation_industry.csv"
        self._api = api
        self.symbols = symbols
        self.fields = ["symbol"] + symbols

    def get_path(self):
        """Get file path for industry relation file"""
        return self._path

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
                if symbols_industries[symbol] == industry:
                    industry_dict[company] = 1
                else:
                    industry_dict[company] = 0
            industry_data.append(industry_dict)

        return industry_data
