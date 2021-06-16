"""Entry point for stock trend prediction"""

from tensorflow import keras
from model.layers import \
    EventInformationEncoder, \
    StockContextEncoder, \
    StockDependentInfluence, \
    StockTrendForecaster, \
    TypeSpecificEncoder


class RESTNet(keras.Model):
    """Architecture for stock trend prediction"""
    def __init__(self):
        super(RESTNet, self).__init__()

        # model architecture
        # trainable
        self.type_specific_encoder = TypeSpecificEncoder()
        self.event_information_encoder = EventInformationEncoder()
        self.stock_context_encoder = StockContextEncoder()
        self.stock_dependent_influence = StockDependentInfluence()
        self.stock_trend_forecaster = StockTrendForecaster()

    def call(self, inputs):
        pass
