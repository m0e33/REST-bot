""" Module for generating data configurations that set the base for the data pipeline"""

from typing import List
import os
from datetime import datetime
from enum import Enum
from configuration.configuration import serialize_cfg, deserialize_cfg

DATA_CFG_CACHING_PATH = "./configuration/cached_data_cfg"


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
        stock_news_limit: int = 500
    ) -> None:
        self.start_str = start
        self.start = datetime.strptime(start, self.DATE_FORMAT).date()

        self.end_str = end
        self.end = datetime.strptime(end, self.DATE_FORMAT).date()
        self.gt_metric = gt_metric
        self.feedback_metrics = feedback_metrics
        self.symbols = symbols
        self.stock_news_limit = stock_news_limit

    """Single source data configuration class"""
    DATE_FORMAT = "%Y-%m-%d"

    def public_method(self):
        """public method 1"""

    def public_method_2(self):
        """public method 2"""

    # For caching to work, we have to define a custom equals method, which checks if values in our config changed.
    def __eq__(self, other):
        return bool(
            self.start_str == other.start_str
            and self.start == other.start
            and self.end_str == other.end_str
            and self.end == other.end
            and self.gt_metric == other.gt_metric
            and self.feedback_metrics == other.feedback_metrics
            and self.symbols == other.symbols
            and self.stock_news_limit == other.stock_news_limit)

    def __repr__(self):
        return str({attr:getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")})


def serialize_data_cfg(cfg: DataConfiguration):
    """ Serializes data fetching configuration using pickle """
    serialize_cfg(DATA_CFG_CACHING_PATH, cfg)


def data_cfg_is_cached():
    """ Checking if data fetching configuration is cached"""
    return os.path.exists(DATA_CFG_CACHING_PATH)


def deserialize_data_cfg():
    """ Deserialize safed data configuration"""
    return deserialize_cfg(DATA_CFG_CACHING_PATH)