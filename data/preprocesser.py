"""Preprocess data for model usage"""
from enum import Enum
import pandas as pd
import numpy as np
import tensorflow as tf
from data.data_store import DataStore
from data.data_configuration import DataConfiguration
from data.data_info import PriceDataInfo


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
        self._events_df = pd.DataFrame()
        self._gt_df = pd.DataFrame()

    def build_events_data_with_gt(self):
        """builds event data"""

        # vertically concatenate all symbols and their events
        self._events_df = pd.concat(
            [self._build_df_for_symbol(symbol) for symbol in self.data_cfg.symbols]
        )

        # join event_title & event_text columns
        self._events_df["event"] = (
            self._events_df["event_type"] + " " + self._events_df["event_text"]
        )
        self._events_df = self._events_df.drop(["event_type", "event_text"], axis=1)
        self._events_df = self._events_df.astype({"event": object})
        # build multi-index dataframe per date and symbol
        #
        # The grouping with gt_trend is unnecessary here, because it holds the same grouping
        # information as 'date'. We have to list it here to copy it over to the new
        # events_df dataframe
        self._events_df = (
            self._events_df.groupby(["date", "symbol", "gt_trend"])["event"]
            .apply(list)
            .reset_index()
        )
        self._events_df.set_index(["date", "symbol"], inplace=True)
        self._events_df.index = self._events_df.index.set_levels(
            self._events_df.index.levels[0].date, level=0
        )

        self._gt_df = self._events_df["gt_trend"]
        self._events_df = self._events_df["event"]

    def get_tf_dataset(self):
        """Return windowed dataset for model based on events_df"""
        dates_count = len(self._events_df.index.levels[0])
        symbols_count = len(self._events_df.index.levels[1])

        # don't ask me..
        np_stock_matrix = self._events_df.values.reshape(dates_count, symbols_count, -1)
        ragged_t = tf.ragged.constant(np_stock_matrix)
        print(ragged_t.shape)
        print("stop")

    def _build_date_dataframe(self):
        dates = pd.date_range(self.data_cfg.start_str, self.data_cfg.end_str, freq="D")
        date_df = pd.DataFrame({"date": dates})
        date_df["date"] = date_df["date"].apply(lambda x: x.date())
        return date_df

    def _build_df_for_symbol(self, symbol):

        symbol_event_df = self._build_events_df_for_symbol(symbol)
        symbol_price_gt_df = self._build_price_gt_df_for_symbol(symbol)

        symbol_df = pd.merge(self.date_df, symbol_event_df, on="date", how="left")
        symbol_df = pd.merge(symbol_df, symbol_price_gt_df, on="date")

        symbol_df["event_type"] = symbol_df["event_type"].replace(
            np.nan, EventType.NO_EVENT.value
        )

        symbol_df["event_text"] = symbol_df["event_text"].replace(
            np.nan, "Nothing happened"
        )

        symbol_df["symbol"] = symbol_df["symbol"].replace(np.nan, symbol)

        return symbol_df

    def _build_events_df_for_symbol(self, symbol):
        symbol_press_df = pd.DataFrame.from_dict(
            self.data_store.get_press_release_data(symbol)
        )
        symbol_press_df = _preprocess_event_df(symbol_press_df, EventType.PRESS_EVENT)

        symbol_news_df = pd.DataFrame.from_dict(
            self.data_store.get_stock_news_data(symbol)
        )
        symbol_news_df = _preprocess_event_df(symbol_news_df, EventType.NEWS_EVENT)

        return pd.concat([symbol_press_df, symbol_news_df], axis=0)

    def _build_price_gt_df_for_symbol(self, symbol):
        symbol_price_df = pd.DataFrame.from_dict(
            self.data_store.get_price_data(symbol),
        )

        symbol_price_df = symbol_price_df.astype(
            {
                "date": str,
                "low": float,
                "high": float,
                "close": float,
                "open": float,
                "vwap": float,
            }
        )

        symbol_price_df["date"] = pd.to_datetime(
            symbol_price_df["date"], format=self.data_cfg.DATE_FORMAT
        ).apply(lambda x: x.date())
        symbol_price_df = pd.merge(
            self.date_df, symbol_price_df, on="date", how="left"
        ).ffill()

        indicator_next_day = (
            symbol_price_df[self.data_cfg.gt_metric.value].shift(1).replace(np.nan, 0)
        )
        indicator_current_day = symbol_price_df[self.data_cfg.gt_metric.value]
        symbol_price_df["gt_trend"] = (
            (indicator_next_day - indicator_current_day) / indicator_current_day * 100
        )

        # only return date and gt label
        return symbol_price_df.drop(
            [field for field in PriceDataInfo.fields if field != "date"], axis=1
        )
