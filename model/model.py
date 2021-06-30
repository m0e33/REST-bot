"""Entry point for stock trend prediction"""

import tensorflow as tf
from tensorflow import keras
from model.layers import (
    EventInformationEncoder,
    StockContextEncoder,
    StockDependentInfluence,
    StockTrendForecaster,
    TypeSpecificEncoder,
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
        self.stock_context_encoder = StockContextEncoder()
        self.stock_dependent_influence = StockDependentInfluence()
        self.stock_trend_forecaster = StockTrendForecaster()

    def call(self, inputs):
        # since we have attached the events feedback to the event embedding
        # we have to extract it here again for the tse to work properly
        events, feedback = self._extract_feedback_and_events(inputs)
        return self.type_specific_encoder(events)

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
                all_values,
            ],
        )

        return events, feedback
