"""Layer"""

from tensorflow import keras


class TypeSpecificEncoder(keras.layers.Layer):
    """Layer docstring"""
    def __init__(self, num_heads):
        super().__init__()
        self._num_heads = num_heads

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""

        self.w = self.add_weight(
            shape=(self._num_heads, input_shape[4], input_shape[3]),
            initializer="random_normal",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self._num_heads, input_shape[3]), initializer="random_normal", trainable=True
        )
        

    def call(self, inputs):
        """The layers forward pass"""
