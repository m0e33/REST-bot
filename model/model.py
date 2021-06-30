"""Entry point for stock trend prediction"""

from tensorflow import keras
from model.layers import \
    EventInformationEncoder, \
    StockContextEncoder, \
    StockDependentInfluence, \
    StockTrendForecaster, \
    TypeSpecificEncoder
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
        return self.type_specific_encoder.call(inputs)
