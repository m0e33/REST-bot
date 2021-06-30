""" Module for generating data configurations that set the base for the data pipeline"""

from typing import List
from datetime import datetime
from enum import Enum


class GroundTruthMetric(Enum):
    """Different technical metrics of a stock from which we can choose to compute the gt"""

    OPEN = "open"
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    VWAP = "vwap"


class DataConfiguration:

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        symbols: List[str],
        start: str,
        end: str,
        feedback_metrics: List[str],
        gt_metric: GroundTruthMetric = GroundTruthMetric.CLOSE,
        stock_context_days: int = 30
    ) -> None:
        self.start_str = start
        self.start = datetime.strptime(start, self.DATE_FORMAT).date()

        self.end_str = end
        self.end = datetime.strptime(end, self.DATE_FORMAT).date()
        self.gt_metric = gt_metric
        self.feedback_metrics = feedback_metrics
        self.symbols = symbols
        self.stock_context_days = stock_context_days

    """Single source data configuration class"""
    # pylint: disable=too-many-arguments

    DATE_FORMAT = "%Y-%m-%d"

    def public_method(self):
        """public method 1"""

    def public_method_2(self):
        """public method 2"""
