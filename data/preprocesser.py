"""Preprocess data for model usage"""
from enum import Enum
import pandas as pd
import numpy as np
from data.data_store import DataStore
from data.data_configuration import DataConfiguration


class EventType(Enum):
    """To distinguish between event types for a stock"""

    PRESS_EVENT = "PRESS"
    NEWS_EVENT = "NEWS"
    NO_EVENT = "NOEVENT"


def _preprocess_event_df(symbol_df, event_type):
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

    def __init__(self, data_store: DataStore, data_cfg: DataConfiguration):
        self.data_store = data_store
        self.data_cfg = data_cfg
        self.date_df = self._build_date_dataframe()

    def build_event_data(self):
        """builds event data"""
        events_df = pd.concat([
            self._build_df_for_symbol(symbol) for symbol in self.data_cfg.symbols])

        events_df['event'] = events_df['event_type'] + " " + events_df['event_text']
        events_df = events_df.drop(['event_type', 'event_text'], axis=1)

        events_df = events_df.groupby(['date', 'symbol'])
        events_df['event'].apply(lambda x: '|'.join(x)).reset_index()
        events_df.set_index(['date', 'symbol'], inplace=True)
        events_df.index = events_df.index.set_levels(events_df.index.levels[0].date, level=0)
        print("holy shit that worked")
        return events_df


    def get_feedback_for_event(self):
        """
        Preprocess feedback data
        """

    def public_method_for_linter(self):
        """
        Public method for linter
        """

    def _build_date_dataframe(self):
        dates = pd.date_range(self.data_cfg.start_str, self.data_cfg.end_str, freq="D")
        date_df = pd.DataFrame({"date": dates})
        date_df["date"] = date_df["date"].apply(lambda x: x.date())
        return date_df

    def _build_df_for_symbol(self, symbol):

        symbol_events_df = self._build_events_df_for_symbol(symbol)
        symbol_price_gt_df = self._build_price_gt_df_for_symbol(symbol)

        symbol_events_df = pd.merge(self.date_df, symbol_events_df, on="date", how="left")

        symbol_events_df["event_type"] = symbol_events_df["event_type"].replace(
            np.nan, EventType.NO_EVENT.value
        )

        symbol_events_df["event_text"] = symbol_events_df["event_text"].replace(
            np.nan, "Nothing happened"
        )

        symbol_events_df["symbol"] = symbol_events_df["symbol"].replace(
            np.nan, symbol
        )

        return symbol_events_df

    def _build_events_df_for_symbol(self, symbol):
        symbol_press_df = pd.DataFrame.from_dict(
            self.data_store.get_press_release_data(symbol)
        )
        symbol_press_df = _preprocess_event_df(
            symbol_press_df, EventType.PRESS_EVENT
        )

        symbol_news_df = pd.DataFrame.from_dict(
            self.data_store.get_stock_news_data(symbol)
        )
        symbol_news_df = _preprocess_event_df(
            symbol_news_df, EventType.NEWS_EVENT
        )

        return pd.concat([symbol_press_df, symbol_news_df], axis=0)

    def _build_price_gt_df_for_symbol(self, symbol):
        symbol_price_df = pd.DataFrame.from_dict(
            self.data_store.get_price_data(symbol)
        )
        print(symbol_price_df.dtypes)
        print("stop")
