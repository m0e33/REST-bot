"""Adapter for fetching relevant data from https://financialmodelingprep.com"""
from typing import Dict, List
import requests as re


def _request(path: str):
    url = APIAdapter.BASE_URL + path + f"apikey={APIAdapter.API_KEY}"
    answer = re.get(url)
    return answer.json()


def get_historical_prices(symbol: str, start: str, end: str) -> List[Dict]:
    """Get historical prices from for given time span"""
    path = "historical-price-full/" + f"{symbol}?from={start}&to={end}&"
    return _request(path)["historical"]


class APIAdapter:
    """Adapter for fetching relevant data from financialmodelingprep.com"""

    API_KEY = "ab94e35951aef133a2befdecb21c20b6"
    BASE_URL = "https://financialmodelingprep.com/api/v3/"

    def __init__(self) -> None:
        pass

    def public_method_for_linter(self):
        """Why do we need a class for APIAdapter?"""

    def another_public_method_for_linter(self):
        """Why do we need a class for APIAdapter?"""
