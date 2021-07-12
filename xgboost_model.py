import logging
from os import path
from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from kubeflow_utils.kubeflow_serve import KubeflowServe
from kubeflow_utils.metadata_config import MetadataConfig
from kubeflow_utils.training_result import TrainingResult

logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)


def eval_model(model, test_X, test_y):
    """Evaluate the model performance."""
    predictions = model.predict(test_X)
    mean_error = mean_absolute_error(predictions, test_y)
    logging.info("mean_absolute_error=%.2f", mean_error)
    return mean_error


class XGBoostModel(KubeflowServe):
    def __init__(self):
        super().__init__()

    def download_data_component(self, cloud_path: str, data_path: str):
        self.download_data(cloud_path, data_path)

    def read_input(self, file_name, test_size=0.5):
        """Read input data and split it into train and test."""
        if not path.exists(file_name):
            self.download_data_component(file_name, './')

        data: pd.Dataframe = pd.read_csv(f'{file_name}/train.csv')
        data.dropna(axis=0, subset=['SalePrice'], inplace=True)

        y = data.SalePrice
        x: pd.Series = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

        train_x, test_x, train_y, test_y = train_test_split(x.values,
                                                            y.values,
                                                            test_size=test_size,
                                                            shuffle=False)
        imputer = SimpleImputer(strategy="median")
        train_x = imputer.fit_transform(train_x)
        test_x = imputer.transform(test_x)

        return (train_x, train_y), (test_x, test_y), imputer

    def get_metadata(self) -> MetadataConfig:
        return MetadataConfig(
            model_names="trained_ames_model.dat",
            model_description="xgboost model for predicting prices",
            model_version="v0.1",
            model_type="linear_regression",
            maturity_state="development",
            dataset_name="ames training data",
            dataset_description="dataset consists of multiple prices",
            dataset_path="ames_dataset",
            dataset_version="v0.3",
            owner="developer@mail.me",
            training_framework_name="xgboost",
            training_framework_version="1.2.1",
        )

    def train_model(self, data_path: str = "ames_dataset", learning_rate=0.1, n_estimators=50, pipeline_run=False) -> TrainingResult:
        (train_X, train_y), (test_X, test_y), _ = self.read_input(data_path)
        """Train the model using XGBRegressor."""
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(train_X, train_y, early_stopping_rounds=40, eval_set=[(test_X, test_y)])

        logging.info("Best RMSE on eval: %.2f with %d rounds",
                     model.best_score,
                     model.best_iteration + 1)

        mae = eval_model(model, test_X, test_y)

        return TrainingResult(
            models=model,
            evaluation={"mean_absolute_error": mae},
            hyperparameters={"n_estimators": n_estimators, "learning_rate": learning_rate}
        )

    def predict_model(self, models, data) -> Union[np.ndarray, List, str, bytes]:
        """Predict using the model for given ndarray."""

        np_array = np.array(data)
        prediction = models[0].predict(np_array)

        return [[prediction.item(0)]]
