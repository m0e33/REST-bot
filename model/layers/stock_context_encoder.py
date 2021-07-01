"""Layer"""
import tensorflow as tf
from tensorflow import keras
from model.configuration import HyperParameterConfiguration
from model.layers.sequence_encoder import SequenceEncoder


class StockContextEncoder(keras.layers.Layer):
    """Layer docstring"""
    def __init__(self, hp_cfg: HyperParameterConfiguration):
        super(StockContextEncoder, self).__init__()

        self.events_sequence_encoder = SequenceEncoder(hp_cfg.sliding_window_size,
                                                       hp_cfg.lstm_units_cnt, from_back=False)
        self.feedback_sequence_encoder = SequenceEncoder(hp_cfg.sliding_window_size, 5,
                                                         from_back=False)

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""


    def call(self, event_embeddings, feedback):
        """The layers forward pass"""
        events_sequence_encodings = self.events_sequence_encoder(event_embeddings)
        feedback_sequence_encodings = self.feedback_sequence_encoder(feedback)

        return tf.concat([events_sequence_encodings, feedback_sequence_encodings], axis=1)
