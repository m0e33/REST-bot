"""Adapter for fetching relevant data from https://financialmodelingprep.com"""
from typing import Dict, List

import requests as re


class APIAdapter:
    """Adapter for fetching relevant data from financialmodelingprep.com"""

    def __init__(self) -> None:
        self.API_KEY = "ab94e35951aef133a2befdecb21c20b6"
        self.BASE_URL = "https://financialmodelingprep.com/api/v3/"

    def get_historical_prices(self, symbol: str, start: str, end: str) -> List[Dict]:
        """Get historical prices from for given time span"""

        path = "historical-price-full/" + f"{symbol}?from={start}&to={end}&"
        return self._request(path)

    def get_press_release_data(self, symbol: str, limit: int) -> List[Dict]:
        """Get press releases for symbol"""

        path = "press-releases/" + f"{symbol}?limit={limit}&"
        return self._request(path)

    def _request(self, path: str):
        url = self.BASE_URL + path + f"apikey={self.API_KEY}"
        answer = re.get(url)
        return answer.json()
