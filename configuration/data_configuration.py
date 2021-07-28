""" Module for generating data configurations that set the base for the data pipeline"""

from typing import List
import os
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from configuration.configuration import serialize_cfg, deserialize_cfg

DATA_CFG_CACHING_PATH = "./cached_data_cfg"


class GroundTruthMetric(Enum):
    """Different technical metrics of a stock from which we can choose to compute the gt"""

    OPEN = "open"
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    VWAP = "vwap"


@dataclass
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


def serialize_data_cfg(cfg: DataConfiguration):
    """ Serializes data fetching configuration using pickle """
    serialize_cfg(DATA_CFG_CACHING_PATH, cfg)


def data_cfg_is_cached():
    """ Checking if data fetching configuration is cached"""
    return os.path.exists(DATA_CFG_CACHING_PATH)


def deserialize_data_cfg():
    """ Deserialize safed data configuration"""
    return deserialize_cfg(DATA_CFG_CACHING_PATH)