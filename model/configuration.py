""" Single source of truth for all sorts of parameters"""
from dataclasses import dataclass


@dataclass
class HyperParameterConfiguration:
    """ Class for storing hyper parameters"""

    attn_cnt = 2
    lstm_units_cnt = 40
    sliding_window_size = 15
    offset_days = 3


@dataclass
class TrainConfiguration:
    """ Class for storing training parameters"""

    val_split = 0.2
    test_split = 0.1
    batch_size = 4

