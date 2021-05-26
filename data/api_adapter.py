"""Adapter for fetching relevant data from https://financialmodelingprep.com"""
from typing import Dict, List

import requests as re


class APIAdapter:
    """Adapter for fetching relevant data from financialmodelingprep.com"""

    def __init__(self) -> None:
        self._key = "ab94e35951aef133a2befdecb21c20b6"
        self.base_url_v3 = "https://financialmodelingprep.com/api/v3/"
        self.base_url_v4 = "https://financialmodelingprep.com/api/v4/"

    def get_historical_prices(self, symbol: str, start: str, end: str) -> List[Dict]:
        """Get historical prices from for given time span"""

        path = "historical-price-full/" + f"{symbol}?from={start}&to={end}&"
        return self._request(path)

    def get_press_releases(self, symbol: str, limit: int) -> List[Dict]:
        """Get press releases for symbol"""

        path = "press-releases/" + f"{symbol}?limit={limit}&"
        return self._request(path)

    def get_industry_classification(self, symbol):
        """Get industrial classification for a symbol, e.g.: 'ELECTRONIC COMPUTERS' for 'AAPL'"""

        path = "standard_industrial_classification/" + f"?symbol={symbol}&"
        return self._request(path, api_version=4)

    def get_stock_peers(self, symbol):
        """Get Stock peers of symbol"""

        path = "stock_peers/" + f"?symbol={symbol}&"
        return self._request(path, api_version=4)

    def get_institutional_holders(self, symbol):
        """Get all institutional holders for a symbol"""

        path = "institutional-holder/" + f"{symbol}?"
        return self._request(path)

    def _request(self, path: str, api_version: int = 3):
        if api_version == 3:
            url = self.base_url_v3 + path + f"apikey={self._key}"
        elif api_version == 4:
            url = self.base_url_v4 + path + f"apikey={self._key}"
        else:
            raise Exception("API Version not available")

        answer = re.get(url)
        return answer.json()
