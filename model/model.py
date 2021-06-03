"""Entry point for stock trend prediction"""

from tensorflow import keras
from model.layers import \
    WordEventEmbeddings, \
    EventInformationEncoder, \
    StockContextEncoder, \
    StockDependentInfluence, \
    StockTrendForecaster


class RESTModel(keras.Model):
    """Architecture for stock trend prediction"""
    def __init__(self):
        super(RESTModel, self).__init__()

        # model architecture
        # preprocessing
        self.word_event_embeddings = WordEventEmbeddings()

        # trainable
        self.event_information_encoder = EventInformationEncoder()
        self.stock_context_encoder = StockContextEncoder()
        self.stock_dependent_influence = StockDependentInfluence()
        self.stock_trend_forecaster = StockTrendForecaster()

    def call(self, inputs):
        pass
