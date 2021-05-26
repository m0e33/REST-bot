from collections import namedtuple
from dataclasses import dataclass
from typing import Any


@dataclass
class PriceDataInfo():
    def __init__(self, base_path):
        self._base_path = base_path
        self._path = f"{self._base_path}prices_{self.start}_{self.end}_"

    start = "2021-01-01"
    end = "2021-04-01"
    fields = ["date", "open", "close", "high", "low", "vwap"]

    def get_path(self, symbol):
        return f"{self._path}{symbol}.csv"

@dataclass
class PressDataInfo():
    def __init__(self, base_path):
        self._base_path = base_path
        self._path = f"{self._base_path}press_limit={self.limit}_"

    limit = 100
    fields = ["symbol", "date", "title", "text"]

    def get_path(self, symbol):
        return f"{self._path}{symbol}.csv"
