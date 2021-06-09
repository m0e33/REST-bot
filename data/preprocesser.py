"""Preprocess data for model usage"""
from enum import Enum
import pandas as pd
import numpy as np
import tensorflow as tf
from data.data_store import DataStore
from data.data_configuration import DataConfiguration
from data.data_info import PriceDataInfo
from model.train_configuration import TrainConfiguration


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

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        data_store: DataStore,
        data_cfg: DataConfiguration,
        train_cfg: TrainConfiguration,
    ):
        self.data_store = data_store
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg
        self.date_df = self._build_date_dataframe()

        # Predefine all dataframes for linter._.
        self._events_train_df = pd.DataFrame()
        self._gt_train_df = pd.DataFrame()
        self._events_val_df = pd.DataFrame()
        self._gt_val_df = pd.DataFrame()
        self._events_test_df = pd.DataFrame()
        self._gt_test_df = pd.DataFrame()

    def build_events_data_with_gt(self):
        """builds event data"""

        # vertically concatenate all symbols and their events
        events_df = pd.concat(
            [self._build_df_for_symbol(symbol) for symbol in self.data_cfg.symbols]
        )

        # join event_title & event_text columns
        events_df["event"] = events_df["event_type"] + " " + events_df["event_text"]
        events_df = events_df.drop(["event_type", "event_text"], axis=1)
        events_df = events_df.astype({"event": object})

        # build multi-index dataframe per date and symbol to later generate tensors
        # with the right shape easily
        #
        # The grouping with gt_trend is unnecessary here, because it holds the same grouping
        # information as 'date' and 'symbol' combined. We have to list it here in order
        # to copy it over to the new events_df dataframe
        events_df = (
            events_df.groupby(["date", "symbol", "gt_trend"])["event"]
            .apply(list)
            .reset_index()
        )
        events_df.set_index(["date", "symbol"], inplace=True)
        events_df.index = events_df.index.set_levels(
            events_df.index.levels[0].date, level=0
        )

        self._set_train_val_test_split(events_df)

    def get_train_ds(self):
        """windowed tensorflow training dataset"""
        return self._get_tf_dataset(self._events_train_df, self._gt_train_df)

    def get_val_ds(self):
        """windowed tensorflow validation dataset"""
        return self._get_tf_dataset(self._events_val_df, self._gt_val_df)

    def get_test_ds(self):
        """windowed tensorflow validation dataset"""
        return self._get_tf_dataset(self._events_test_df, self._gt_test_df)

    def _get_tf_dataset(self, events_df, gt_df):
        """Return windowed dataset based on events_df and ground truth"""

        sliding_window_length = self.data_cfg.stock_context_days

        dates_count = len(events_df.groupby(level=0))
        symbols_count = len(events_df.groupby(level=1))

        # build the input tensor
        # we have to use ragged tensor here, since we don't know how many events
        # per date and symbol there are
        np_stock_matrix = events_df.values.reshape(dates_count, symbols_count, 1)
        events_ragged_t = tf.ragged.constant(np_stock_matrix)
        events_ragged_t = tf.squeeze(events_ragged_t, axis=[2])

        # timeseries_dataset_from_array only takes eager tensors with defined shape.
        # The last dimension of the ragged tensor is padded to match the longest
        # element in this dimension
        events_t = events_ragged_t.to_tensor()

        # build the output tensor
        np_gt_trend_matrix = gt_df.values.reshape(dates_count, symbols_count, 1)

        # since the 'timeseries_dataset_from_array' documentation states:
        #
        # "targets[i] should be the target corresponding to the window that starts at index i"
        #
        # we have to 'shift' the gt_tensor #{sliding_window_length} time steps 'back in time',
        # so that target[1] yields the gt for the first window, which otherwise would be at
        # target[{sliding_window_length}]
        np_gt_trend_matrix = np.roll(
            np_gt_trend_matrix, shift=sliding_window_length, axis=0
        )

        gt_trend_t = tf.constant(np_gt_trend_matrix)

        return tf.keras.preprocessing.timeseries_dataset_from_array(
            data=events_t,
            targets=gt_trend_t,
            sequence_length=sliding_window_length,
            sequence_stride=1,
            batch_size=32,
        )

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

    def _set_train_val_test_split(self, events_df):

        actual_val_split = 1 - (self.train_cfg.val_split + self.train_cfg.test_split)
        actual_test_split = 1 - self.train_cfg.test_split

        # since np.split does not take hierarchical indexing into account
        # but rather flattens the index we have to make sure not to split
        # in the middle of a day
        dates_count = len(events_df.index.levels[0])
        symbols_count = len(events_df.index.levels[1])

        dates_val_split = int(dates_count * actual_val_split) * symbols_count
        dates_test_split = int(dates_count * actual_test_split) * symbols_count

        # pylint: disable=unbalanced-tuple-unpacking
        events_train_df, events_val_df, events_test_df = np.split(
            events_df,
            [
                dates_val_split,
                dates_test_split,
            ],
        )

        self._events_train_df = events_train_df["event"]
        self._gt_train_df = events_train_df["gt_trend"]

        self._events_val_df = events_val_df["event"]
        self._gt_val_df = events_val_df["gt_trend"]

        self._events_test_df = events_test_df["event"]
        self._gt_test_df = events_test_df["gt_trend"]
