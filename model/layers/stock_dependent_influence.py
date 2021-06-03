"""Layer"""

from tensorflow import keras


class StockDependentInfluence(keras.layers.Layer):
    """Layer docstring"""
    def __init__(self):
        super(StockDependentInfluence, self).__init__()

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""

    def call(self, inputs):
        """The layers forward pass"""
