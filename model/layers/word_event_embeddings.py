"""Layer"""

from tensorflow import keras


class WordEventEmbeddings(keras.layers.Layer):
    """Layer docstring"""

    def __init__(self, embedding_dim=100):
        super(WordEventEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        """Gets executed, the first time the layer gets called"""

    def call(self, inputs):
        """The layers forward pass"""
