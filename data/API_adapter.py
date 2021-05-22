from typing import Dict, List
import requests as re
import json


class APIAdapter:
    API_KEY = "ab94e35951aef133a2befdecb21c20b6"
    BASE_URL = "https://financialmodelingprep.com/api/v3/"

    def __init__(self) -> None:
        pass

    def _request(self, path: str):
        url = APIAdapter.BASE_URL + path + f"apikey={APIAdapter.API_KEY}"
        answer = re.get(url)

        return answer.json()

    def get_historical_prices(self, symbol: str, start: str, end: str) -> List[Dict]:
        path = "historical-price-full/" + f"{symbol}?from={start}&to={end}&"
        return self._request(path)["historical"]
