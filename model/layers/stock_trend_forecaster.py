"""Layer"""

from tensorflow import keras


class StockTrendForecaster(keras.layers.Layer):
    """Layer docstring"""
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""

    def call(self, inputs):
        """The layers forward pass"""
