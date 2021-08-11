"""Layer"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Layer


class StockDependentInfluence(Layer):
    """Layer docstring"""
    def __init__(self):
        super().__init__()

        self.dense = None

    def build(self, inputs_shape):
        """Gets executed, the first time the layer gets called"""
        # will be 40 (lstm hidden units count)
        units = inputs_shape[0][1]
        self.dense = Dense(units, activation=LeakyReLU())

    def call(self, inputs):
        """The layers forward pass"""
        last_events, stock_context = inputs

        concat = tf.concat([last_events, stock_context], axis=1)
        effect_strength = self.dense(concat)

        effect_of_event_information = effect_strength * last_events

        return effect_of_event_information
