"""Preprocess data for model usage"""
import logging
import pickle
from enum import Enum
from pathlib import Path
import os.path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from data.data_store import DataStore
from configuration.data_configuration import DataConfiguration
from data.data_info import PriceDataInfo
from configuration.configuration import (
    TrainConfiguration,
    HyperParameterConfiguration,
    hp_cfg_is_cached,
    deserialize_hp_cfg,
    serialize_hp_cfg,
    train_cfg_is_cached,
    deserialize_train_cfg,
    serialize_train_cfg,
)

logger = logging.getLogger("preprocessor")


class EventType(Enum):
    """To distinguish between event types for a stock"""

    PRESS_EVENT = "PRESS"
    NEWS_EVENT = "NEWS"
    NO_EVENT = "NOEVENT"


class DatasetType(Enum):
    """Dataset Enum"""

    TRAIN_DS = "train"
    VAL_DS = "val"
    TEST_DS = "test"


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


def store_dataset_np_to_file_system(events_df, gt_df, base_dataset_path):
    Path(base_dataset_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(events_df, open(base_dataset_path + "events.p", "wb"))
    pickle.dump(gt_df, open(base_dataset_path + "gt.p", "wb"))


def restore_dataset_df_from_file(base_dataset_path):
    events_df = pickle.load(open(base_dataset_path + "events.p", "rb"))
    gt_df = pickle.load(open(base_dataset_path + "gt.p", "rb"))

    return [events_df, gt_df]


def dataset_np_has_been_build(base_path_to_ds):
    return os.path.isfile(base_path_to_ds + "events.p") and os.path.isfile(
        base_path_to_ds + "gt.p"
    )


class Preprocessor:

    """Preprocess data for model usage"""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        data_store: DataStore,
        data_cfg: DataConfiguration,
        train_cfg: TrainConfiguration,
        hp_cfg: HyperParameterConfiguration,
    ):
        self.data_store = data_store
        self.data_cfg = data_cfg

        assert (
            len(set(self.data_cfg.feedback_metrics) - set(PriceDataInfo.fields)) == 0
        ), "API data price fields do not contain all fields that are configured as feedback metrics"

        self.train_cfg = train_cfg
        self.hp_cfg = hp_cfg

        # advanced caching mechanism needs to safe new configurations
        self._old_preprocessing_result_can_be_reused = (
            self._check_reusability_of_old_preprocessing()
        )
        logger.info(
            "Preprocessing result reusable: "
            + str(self._old_preprocessing_result_can_be_reused)
        )
        serialize_hp_cfg(self.hp_cfg)
        serialize_train_cfg(self.train_cfg)

        self.date_df = self._build_date_dataframe()

        self.train_ds_generator_spec = []
        self.val_ds_generator_spec = []
        self.test_ds_generator_spec = []

        self._vectorizer = TextVectorization(
            max_tokens=self.data_cfg.stock_news_limit,
            output_sequence_length=self.MAX_EVENT_LENGTH,
        )
        self.embedding_model: Sequential
        self._prepare_word_embedding()

    NOTHING_HAPPENED_TEXT = "Nothing happened"
    EMBEDDING_DIM = 300
    MAX_EVENT_LENGTH = 50
    PATH_TO_GLOVE_FILE = "data/assets/glove.6B.300d.txt"
    PATH_FOR_TRAIN_NP = "data/datasets/train/"
    PATH_FOR_VAL_NP = "data/datasets/val/"
    PATH_FOR_TEST_NP = "data/datasets/test/"

    def build_events_data_with_gt(self):
        """builds event data"""

        # check cached events_df
        if self._old_preprocessing_result_can_be_reused:
            return

        # vertically concatenate all symbols and their events
        events_df = pd.concat(
            [self._build_df_for_symbol(symbol) for symbol in self.data_cfg.symbols]
        )

        # join event_title & event_text columns
        events_df["event"] = events_df["event_type"] + " " + events_df["event_text"]
        events_df = events_df.drop(["event_type", "event_text"], axis=1)
        events_df = events_df.astype({"event": object})

        # We have to incorporate the feedback at the end of the event embedding vector
        # pylint: disable=unnecessary-lambda
        events_df["event"] = events_df.apply(
            lambda row: self._create_embedding_with_feedback(row), axis=1
        )

        # We incorporated the feedback at the end of the event embedding matrix
        # so we don't need the single metrics anymore
        events_df = events_df.drop(self.data_cfg.feedback_metrics, axis=1)

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

        [
            events_train_df,
            gt_train_df,
            events_val_df,
            gt_val_df,
            events_test_df,
            gt_test_df,
        ] = self._get_train_val_test_split(events_df)

        np_input_train, np_gt_train = self._convert_to_tf_compliant_np_matrices(
            events_train_df, gt_train_df, DatasetType.TRAIN_DS
        )
        np_input_val, np_gt_val = self._convert_to_tf_compliant_np_matrices(
            events_val_df, gt_val_df, DatasetType.VAL_DS
        )
        np_input_test, np_gt_test = self._convert_to_tf_compliant_np_matrices(
            events_test_df, gt_test_df, DatasetType.TEST_DS
        )

        store_dataset_np_to_file_system(
            np_input_train, np_gt_train, self.PATH_FOR_TRAIN_NP
        )
        store_dataset_np_to_file_system(np_input_val, np_gt_val, self.PATH_FOR_VAL_NP)
        store_dataset_np_to_file_system(
            np_input_test, np_gt_test, self.PATH_FOR_TEST_NP
        )

    def get_train_ds(self):
        """windowed tensorflow training dataset"""
        assert dataset_np_has_been_build(
            self.PATH_FOR_TRAIN_NP
        ), "Train dataset numpy array has not yet been build"
        return tf.data.Dataset.from_generator(
            self._dataset_generator,
            args=[self.PATH_FOR_TRAIN_NP],
            output_signature=self.train_ds_generator_spec,
        )

    def get_val_ds(self):
        """windowed tensorflow validation dataset"""
        pass

    def get_test_ds(self):
        """windowed tensorflow test dataset"""
        pass

    def _dataset_generator(self, base_ds_path):
        print(base_ds_path)

    def _prepare_word_embedding(self):
        if self._old_preprocessing_result_can_be_reused:
            return

        self._set_vectorizer()

        vocab = self._vectorizer.get_vocabulary()
        num_tokens = len(vocab) + 2

        embedding_matrix = self._build_embedding_matrix(vocab)
        embedding = Embedding(
            num_tokens,
            self.EMBEDDING_DIM,
            input_length=self.MAX_EVENT_LENGTH,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )

        self.embedding_model = Sequential()
        self.embedding_model.add(embedding)
        self.embedding_model.compile()

    def _build_date_dataframe(self):
        dates = pd.date_range(self.data_cfg.start_str, self.data_cfg.end_str, freq="D")
        date_df = pd.DataFrame({"date": dates})
        date_df["date"] = date_df["date"].apply(lambda x: x.date())
        return date_df

    def _build_df_for_symbol(self, symbol):

        symbol_event_df = self._build_events_df_for_symbol(symbol)
        symbol_feedback_and_gt_df = self._build_price_gt_df_for_symbol(symbol)

        symbol_df = pd.merge(self.date_df, symbol_event_df, on="date", how="left")
        symbol_df = pd.merge(symbol_df, symbol_feedback_and_gt_df, on="date")

        symbol_df["event_type"] = symbol_df["event_type"].replace(
            np.nan, EventType.NO_EVENT.value
        )

        symbol_df["event_text"] = symbol_df["event_text"].replace(
            np.nan, self.NOTHING_HAPPENED_TEXT
        )

        symbol_df["symbol"] = symbol_df["symbol"].replace(np.nan, symbol)

        return symbol_df

    def _build_events_df_for_symbol(self, symbol):
        symbol_press_df = self._get_symbol_press_df(symbol)
        symbol_news_df = self._get_symbol_news_df(symbol)

        return pd.concat([symbol_press_df, symbol_news_df], axis=0)

    def _get_symbol_press_df(self, symbol):
        symbol_press_df = pd.DataFrame.from_dict(
            self.data_store.get_press_release_data(symbol)
        )

        return _preprocess_event_df(symbol_press_df, EventType.PRESS_EVENT)

    def _get_symbol_news_df(self, symbol):
        symbol_news_df = pd.DataFrame.from_dict(
            self.data_store.get_stock_news_data(symbol)
        )

        return _preprocess_event_df(symbol_news_df, EventType.NEWS_EVENT)

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

        symbol_feedback_df = symbol_price_df.drop(["date"], axis=1)

        indicator_next_day = symbol_feedback_df.shift(-1).replace(np.nan, 0)
        indicator_current_day = symbol_feedback_df
        symbol_feedback_df = (
            indicator_next_day - indicator_current_day
        ) / indicator_current_day

        symbol_feedback_df = symbol_feedback_df.join(symbol_price_df["date"])

        # duplicate symbols gt metric column with dedicated gt label
        symbol_feedback_df["gt_trend"] = symbol_feedback_df[
            self.data_cfg.gt_metric.value
        ]

        # return all fields which are choosen for feedback metrics and gt
        return symbol_feedback_df.drop(
            [
                field
                for field in PriceDataInfo.fields
                if field != "date"
                and field != "gt_trend"
                and field not in self.data_cfg.feedback_metrics
            ],
            axis=1,
        )

    def _get_train_val_test_split(self, events_df):

        actual_val_split = 1 - (self.train_cfg.val_split + self.train_cfg.test_split)
        actual_test_split = 1 - self.train_cfg.test_split

        # since np.split does not take hierarchical indexing into account
        # but rather flattens the index, we have to make sure not to split
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

        return [
            events_train_df["event"],
            events_train_df["gt_trend"],
            events_val_df["event"],
            events_val_df["gt_trend"],
            events_test_df["event"],
            events_test_df["gt_trend"],
        ]

    def _get_event_texts_for_symbol(self, symbol):
        press_texts = self._get_symbol_press_df(symbol)["event_text"]
        press_texts = EventType.PRESS_EVENT.value + " " + press_texts
        news_texts = self._get_symbol_news_df(symbol)["event_text"]
        news_texts = EventType.NEWS_EVENT.value + " " + news_texts

        return pd.concat([press_texts, news_texts], axis=0)

    def _create_embedding_with_feedback(self, events_df_row):
        event_string = events_df_row["event"]
        event_vector = self._vectorizer([event_string])
        event_embedding = self.embedding_model.predict(event_vector)

        # event embedding comes in the shape [1,50,300], we want the shape [50, 300],
        # which represents one sentence much better.
        event_embedding = np.squeeze(event_embedding)

        # we have to append the events feedback to the event embedding keep the dataset
        # shape working therefore each feedback metric has to be expressed with a
        # (300) vector.
        feedback_row = events_df_row[self.data_cfg.feedback_metrics].values
        new_feedback_shape = (len(self.data_cfg.feedback_metrics), self.EMBEDDING_DIM)
        feedback_row = np.broadcast_to(
            np.expand_dims(feedback_row, axis=1), new_feedback_shape
        )

        return np.concatenate((event_embedding, feedback_row), axis=0)

    def _set_vectorizer(self):
        all_event_texts = pd.concat(
            [
                self._get_event_texts_for_symbol(symbol)
                for symbol in self.data_cfg.symbols
            ]
        )
        all_event_texts = all_event_texts.append(
            pd.Series(EventType.NO_EVENT.value + " " + self.NOTHING_HAPPENED_TEXT)
        )

        self._vectorizer.adapt(
            tf.data.Dataset.from_tensor_slices(all_event_texts.values).batch(128)
        )

    def _build_embedding_matrix(self, vocab):
        # setup word index
        word_index = dict(zip(vocab, range(len(vocab))))

        # setup embedding index
        embeddings_index = {}
        with open(self.PATH_TO_GLOVE_FILE) as file:
            for line in file:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        hits = 0
        misses = 0
        # construct embedding matrix
        missed_words = []
        num_tokens = len(vocab) + 2
        embedding_matrix = np.zeros((num_tokens, self.EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                missed_words.append(word)
                misses += 1
        logger.info("Converted %d words (%d misses)" % (hits, misses))

        return embedding_matrix

    def _check_reusability_of_old_preprocessing(self):
        np_dataset_matrices_have_been_build = bool(
            dataset_np_has_been_build(self.PATH_FOR_TRAIN_NP)
            and dataset_np_has_been_build(self.PATH_FOR_VAL_NP)
            and dataset_np_has_been_build(self.PATH_FOR_TEST_NP)
        )

        new_configs = self._hp_cfg_has_changed() or self._train_cfg_has_changed()

        return (
            np_dataset_matrices_have_been_build
            and not new_configs
            and self.data_store.old_data_can_be_reused
        )

    def _hp_cfg_has_changed(self):
        if hp_cfg_is_cached():
            old_cfg = deserialize_hp_cfg()
            return old_cfg != self.hp_cfg
        return True

    def _train_cfg_has_changed(self):
        if train_cfg_is_cached():
            old_cfg = deserialize_train_cfg()
            return old_cfg != self.train_cfg
        return True

    def _convert_to_tf_compliant_np_matrices(
        self, events_df, gt_df, dataset_type: DatasetType
    ):

        sliding_window_length = self.hp_cfg.sliding_window_size

        dates_count = len(events_df.groupby(level=0))
        symbols_count = len(events_df.groupby(level=1))

        assert sliding_window_length < dates_count, (
            f"sliding window length ({sliding_window_length}) "
            f"does exceed date count ({dates_count}) in dataset."
        )

        # build the input np matrix

        np_stock_matrix = events_df.values.reshape(dates_count, symbols_count, 1)

        events_counts = []

        def add_to_events_counts(list_input):
            events_counts.append(len(list_input[0]))

        np.apply_along_axis(add_to_events_counts, axis=2, arr=np_stock_matrix)

        max_event_count = max(events_counts)

        # tensorflow datasets only takes np arrays with defined shape.
        # The third to last dimension of the np stock array (events count) is padded
        # to match the longest element in this dimension

        def array_cast(list_input):
            unfold_event_list = np.asarray(list_input[0])
            return np.pad(
                unfold_event_list,
                (
                    (0, max_event_count - unfold_event_list.shape[0]),
                    (0, 0),
                    (0, 0),
                ),
            )

        np_stock_matrix = np.apply_along_axis(array_cast, axis=2, arr=np_stock_matrix)

        # build the gt np matrix
        np_gt_trend_matrix = gt_df.values.reshape(dates_count, symbols_count, 1)

        # we have to 'shift' the gt_tensor #{sliding_window_length} time steps 'back in time',
        # so that np_gt_trend_matrix[0] yields the gt for the np_stock_matrix[0] first window,
        # which otherwise would be at np_gt_trend_matrix[{sliding_window_length}]
        np_gt_trend_matrix = np.roll(
            np_gt_trend_matrix, shift=-(sliding_window_length - 1), axis=0
        )

        # The 'from_generator' method needs a spec desription of the tensors yielded by the generator method.
        # This description can be crafted on the fly within this method.
        input_spec = tf.TensorSpec(
            shape=(
                sliding_window_length,
                np_stock_matrix.shape[1],
                np_stock_matrix.shape[2],
                np_stock_matrix.shape[3],
                np_stock_matrix.shape[4],
            ),
            dtype=tf.float16,
        )

        gt_spec = tf.TensorSpec(
            shape=(1, np_gt_trend_matrix.shape[1], np_gt_trend_matrix.shape[2]),
            dtype=tf.float16,
        )

        if dataset_type is DatasetType.TRAIN_DS:
            self.train_ds_generator_spec = (input_spec, gt_spec)
        if dataset_type is DatasetType.VAL_DS:
            self.val_ds_generator_spec = (input_spec, gt_spec)
        if dataset_type is DatasetType.TEST_DS:
            self.test_ds_generator_spec = (input_spec, gt_spec)

        return [np_stock_matrix, np_gt_trend_matrix]
