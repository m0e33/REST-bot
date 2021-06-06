"""Preprocess data for model usage"""
from enum import Enum
import pandas as pd
import numpy as np
from data.data_store import DataStore


class EventType(Enum):
    """To distinguish between event types for a stock"""

    PRESS_EVENT = "PRESS"
    NEWS_EVENT = "NEWS"
    NO_EVENT = "NOEVENT"


def _preprocess_symbol_event_df(symbol_df, event_type):
    if event_type == EventType.NEWS_EVENT:
        symbol_df["date"] = pd.to_datetime(symbol_df["publishedDate"])
        symbol_df.drop(["publishedDate", "site", "url"], axis=1, inplace=True)
    else:
        symbol_df["date"] = pd.to_datetime(symbol_df["date"])

    symbol_df["date"] = symbol_df["date"].apply(lambda x: x.date())
    symbol_df["event_type"] = event_type.value
    symbol_df["event_text"] = symbol_df["title"] + " " + symbol_df["text"]

    return symbol_df.drop(["title", "text"], axis=1)


class Preprocessor:

    """Preprocess data for model usage"""

    def __init__(self, data_store: DataStore):
        self.data_store = data_store
        self.date_df = self._build_date_dataframe()

    def build_event_data(self):
        """builds event data"""

        apple_press_df = pd.DataFrame.from_dict(
            self.data_store.get_press_release_data("AAPL")
        )
        apple_press_df = _preprocess_symbol_event_df(
            apple_press_df, EventType.PRESS_EVENT
        )

        apple_news_df = pd.DataFrame.from_dict(
            self.data_store.get_stock_news_data("AAPL")
        )
        apple_news_df = _preprocess_symbol_event_df(
            apple_news_df, EventType.NEWS_EVENT
        )
        apple_df = pd.concat([apple_press_df, apple_news_df], axis=0)
        apple_df["event_type"] = apple_df["event_type"].replace(
            np.nan, EventType.NO_EVENT.value
        )

        apple_df = pd.merge(self.date_df, apple_df, on="date", how="left")

        apple_df["event_type"] = apple_df["event_type"].replace(
            np.nan, EventType.NO_EVENT.value
        )
        apple_df["event_text"] = apple_df["event_text"].replace(
            np.nan, "Nothing happened"
        )

        apple_df["symbol"] = apple_df["symbol"].replace(
            np.nan, "AAPL"
        )

    def get_feedback_for_event(self):
        """
        Preprocess feedback data
        """

    def public_method_for_linter(self):
        """
        Public method for linter
        """

    def _build_date_dataframe(self):
        dates = pd.date_range(self.data_store.start, self.data_store.end, freq="D")
        date_df = pd.DataFrame({"date": dates})
        date_df["date"] = date_df["date"].apply(lambda x: x.date())
        return date_df
