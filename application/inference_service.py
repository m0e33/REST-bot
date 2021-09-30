from data.data_store import DataStore
import os
from datetime import datetime, timedelta

from model.rest_kubeflow_adapter import KubeflowAdapter
from configuration.data_configuration import deserialize_data_cfg, DataConfiguration
from configuration.configuration import deserialize_hp_cfg, deserialize_train_cfg
from data.preprocesser import Preprocessor

class InferenceEnvironmentException(Exception):
    pass

class InferenceService:
    """Class that encapsulates prediction"""

    def __init__(self):
        self._environment_check()
        self.model = KubeflowAdapter(num_gpus=0, for_inference=True)
        self.hp_cfg = deserialize_hp_cfg()
        self.train_cfg = deserialize_train_cfg()
        self.data_cfg_training = deserialize_data_cfg()

    def get_prediction(self, symbols: list, date: str):

        # Check that symbols have been viewed before
        assert set(symbols) <= set(self.data_cfg_training.symbols)

        # input shape consists of (days, symbols, events, words (+feedback), token representation)
        input = self._build_input_tensor(symbols, date)

        # Model needs batch as outer most dimension, hence single array brackets
        prediction = self.model.predict([input])

        result = {}
        for idx, symbol in enumerate(symbols):
            result[symbol] = prediction[idx][0]

        return result

    def _build_input_tensor(self, symbols: list, end_date_str: str):

        data_cfg_inference = DataConfiguration(
            symbols=symbols,
            start=self._get_start_date(end_date_str),
            end=end_date_str,
            feedback_metrics=["open", "close", "high", "low", "vwap"],
            # this setting assumes that the inference starts at a date close to today
            stock_news_fetch_limit=100,
            events_per_day_limit=self.data_cfg_training.events_per_day_limit
        )

        ds_inference = DataStore(data_cfg_inference, for_inference=True)
        ds_inference.build()
        prepro_inference = Preprocessor(ds_inference, data_cfg_inference, self.train_cfg, self.hp_cfg,
                                        for_inference=True)
        input = prepro_inference.get_inference_input()

        return input

    def _get_start_date(self, end_date: str):
        end_date = datetime.strptime(end_date, self.data_cfg_training.DATE_FORMAT)
        start_date = end_date - timedelta(days=self.hp_cfg.sliding_window_size)
        start_date_str = datetime.strftime(start_date, self.data_cfg_training.DATE_FORMAT)

        return start_date_str

    def _environment_check(self):
        try:
            deserialize_data_cfg()
        except Exception as e:
            raise InferenceEnvironmentException("No data configuration stored on alongside model")

        try:
            deserialize_hp_cfg()
        except Exception as e:
            raise InferenceEnvironmentException("No hyperparameter configuration stored on alongside model")

        try:
            deserialize_train_cfg()
        except Exception as e:
            raise InferenceEnvironmentException("No training configuration stored on alongside model")

        if len(os.listdir(Preprocessor.EMBEDDING_MODEL_CACHE_PATH)) == 0:
            raise InferenceEnvironmentException("No word embedding model stored on alongside model")

        if not os.path.exists("model/weights/RESTNet.index"):
            raise InferenceEnvironmentException("No model stored for inference")