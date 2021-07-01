"""Entry point for stock trend prediction"""

import tensorflow as tf
from tensorflow import keras
from model.layers import (
    StockContextEncoder,
    StockDependentInfluence,
    StockTrendForecaster,
    TypeSpecificEncoder,
    SequenceEncoder
)
from model.configuration import HyperParameterConfiguration


class RESTNet(keras.Model):
    """Architecture for stock trend prediction"""

    def __init__(self, hp_cfg: HyperParameterConfiguration):
        super(RESTNet, self).__init__()

        # parameter
        self.hp_cfg = hp_cfg

        # model architecture
        self.type_specific_encoder = TypeSpecificEncoder(self.hp_cfg.attn_cnt)

        # event information encoder
        self.event_sequence_encoder = SequenceEncoder(self.hp_cfg.offset_days, self.hp_cfg.lstm_units_cnt)

        # stock context encoder
        self.stock_context_encoder = StockContextEncoder(self.hp_cfg)
        self.stock_dependent_influence = StockDependentInfluence()
        self.stock_trend_forecaster = StockTrendForecaster()

    def call(self, inputs):
        # since we have attached the events feedback to the event embedding
        # we have to extract it here again for the tse to work properly
        events, feedback = self._extract_feedback_and_events(inputs)
        event_embeddings = self.type_specific_encoder(events)
        last_events_sequence_encoding = self.event_sequence_encoder(event_embeddings)
        stock_context = self.stock_context_encoder(event_embeddings, feedback)

        # we have to input a list here for build shape extraction to work
        effect_of_event_information = self.stock_dependent_influence([last_events_sequence_encoding, stock_context])

        return last_events_sequence_encoding

    def _extract_feedback_and_events(self, input):

        all_days = input.shape[0]
        all_symbols = input.shape[1]
        all_events = input.shape[2]
        event_word_embeddings = input.shape[3] - 5
        all_values = input.shape[4]
        feedback_metrics_embeddings = 5

        events = tf.slice(
            input,
            begin=[0, 0, 0, 0, 0],
            size=[all_days, all_symbols, all_events, event_word_embeddings, all_values],
        )

        feedback = tf.slice(
            input,
            begin=[0, 0, 0, event_word_embeddings, 0],
            size=[
                all_days,
                all_symbols,
                all_events,
                feedback_metrics_embeddings,
                1,
            ],
        )

        feedback = tf.squeeze(feedback)

        return events, feedback
