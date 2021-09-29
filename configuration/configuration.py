""" Single source of truth for all sorts of parameters"""
import pickle
import os
from dataclasses import dataclass

HP_CACHING_PATH = "./configuration/cached_hp_cfg"
TRAIN_CACHING_PATH = "./configuration/cached_train_cfg"


@dataclass
class HyperParameterConfiguration:
    """ Class for storing hyper parameters"""

    num_epochs: int = 10
    attn_cnt: int = 4
    lstm_units_cnt: int = 80
    sliding_window_size: int = 30
    offset_days: int = 3


@dataclass  # type: ignore
class TrainConfiguration:
    """ Class for storing training parameters"""
    val_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 8


def serialize_hp_cfg(cfg: HyperParameterConfiguration):
    """ Serializes hyper parameter configuration """
    return serialize_cfg(HP_CACHING_PATH, cfg)


def hp_cfg_is_cached():
    """ Checking if hyper parameter configuration is cached"""
    return os.path.exists(HP_CACHING_PATH)


def deserialize_hp_cfg():
    """ Deserialize safed config with pickle"""
    return deserialize_cfg(HP_CACHING_PATH)


def serialize_train_cfg(cfg: TrainConfiguration):
    """ Serializes hyper parameter configuration"""
    serialize_cfg(TRAIN_CACHING_PATH, cfg)


def train_cfg_is_cached():
    """ Checking if train configuration is cached"""
    return os.path.exists(TRAIN_CACHING_PATH)


def deserialize_train_cfg():
    """ Deserialize safed config with pickle"""
    return deserialize_cfg(TRAIN_CACHING_PATH)


def serialize_cfg(path: str, cfg):
    """Serialize any config"""
    with open(path, "wb") as f:
        pickle.dump(cfg, f)


def deserialize_cfg(path: str):
    """ Deserialize any config"""
    with open(path, "rb") as f:
        return pickle.load(f)
