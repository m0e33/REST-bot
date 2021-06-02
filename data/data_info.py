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
        return self._api.get_press_releases(symbol, self.limit)


class StockNewsDataInfo:
    """Information for Stock News Data"""

    def __init__(self, base_path, api: APIAdapter):
        self._base_path = base_path
        self._path = f"{self._base_path}stock_news_limit={self.limit}_"
        self._api = api

    limit = 100
    fields = ["symbol", "publishedDate", "title", "text", "site", "url"]

    def get_path(self, symbol):
        """File path for press release data"""
        return f"{self._path}{symbol}.csv"

    def get_data(self, symbol: str):
        """Get press release data via api"""
        return self._api.get_stock_news(symbol, self.limit)


class IndustryRelationDataInfo:
    """Information for representing industry relation between symbols"""

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
                industry_dict[company] = (
                    1 if symbols_industries[symbol] == industry else 0
                )
            industry_data.append(industry_dict)

        return industry_data


class StockPeerRelationDataInfo:
    """Information to represent stock peer relations between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        self._base_path = base_path
        self._path = f"{self._base_path}relation_peers.csv"
        self._api = api
        self.symbols = symbols
        self.fields = ["symbol"] + symbols

    def get_path(self):
        """Get file path for industry relation file"""
        return self._path

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


class InstitutionalHoldersRelationDataInfo:
    """Information to represent institutional holder relations between symbols"""

    def __init__(self, base_path, api: APIAdapter, symbols: List[str]):
        self._base_path = base_path
        self._path = f"{self._base_path}relation_instholders.csv"
        self._api = api
        self.symbols = symbols
        self.fields = ["symbol"] + symbols

    def get_path(self):
        """Get file path for industry relation file"""
        return self._path

    def get_data(self):
        """Get institutional holders relation of symbols via api"""

        companies = list(
            filter(
                None,
                [self._api.get_institutional_holders(symbol) for symbol in self.fields],
            )
        )

        companies_sorted_and_stripped = [
            sorted(holders, key=lambda holder: int(holder["shares"]), reverse=True)[:10]
            for holders in companies
        ]

        company_holders = {}
        for idx, holders in enumerate(companies_sorted_and_stripped):
            company_holders[self.symbols[idx]] = [
                holder["holder"] for holder in holders
            ]

        holder_data = []
        for symbol in self.symbols:
            holder_dict = {}
            holder_dict["symbol"] = symbol
            for company, holders in company_holders.items():
                if bool(set(holders) & set(company_holders[symbol])):
                    holder_dict[company] = 1
                else:
                    holder_dict[company] = 0
            holder_data.append(holder_dict)

        return holder_data
