"""Layer"""

from tensorflow import keras
from tensorflow.keras.layers import LSTM
import tensorflow as tf


class EventSequenceEncoder(keras.layers.Layer):
    """Layer docstring"""
    def __init__(self, num_days):
        super().__init__()
        self._num_days = num_days

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""
        self._lstm = LSTM(input_shape[-1])


    def call(self, inputs):
        """The layers forward pass"""

        # input shape is (dates, symbols, events, attention_vals)
        # we need the last num_days only
        # -> (num_days, symbols, events, attention_vals)

        # LSTM takes input shape of (batch_size, timesteps, features) where batch_size are the number of symbols in our case
        # we need to switch the first two dimensions of our tensor
        # -> (symbols, num_days, events, attention_vals)

        # each event is a timestep so we need to flatten dimensions 1 and 2
        # -> (symbols, num_days * events, attention_vals)

        data_last_days = tf.slice(inputs, begin=[inputs.shape[0]-self._num_days, 0, 0, 0], size=[self._num_days, inputs.shape[1], inputs.shape[2], inputs.shape[3]])
        transposed_days = tf.transpose(data_last_days, perm=[1, 0, 2, 3])
        flattend_days = tf.reshape(transposed_days, shape=[transposed_days.shape[0], transposed_days.shape[1] * transposed_days.shape[2], transposed_days.shape[3]])
        symbol_representations = self._lstm(flattend_days)

        return symbol_representations
