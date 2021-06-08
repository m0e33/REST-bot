""" Module for generating data configurations that set the base for the data pipeline"""

from typing import List
from datetime import datetime


class DataConfiguration:
    """ Data configuration class"""

    def __init__(self, symbols: List[str], start: str, end: str) -> None:
        self.start_str = start
        self.start = datetime.strptime(start, self.DATE_FORMAT).date()

        self.end_str = end
        self.end = datetime.strptime(end, self.DATE_FORMAT).date()

        self.symbols = symbols

    DATE_FORMAT = "%Y-%m-%d"

    def public_method(self):
        """public method 1"""

    def public_method_2(self):
        """public method 2"""
