import sys
import os
# sys.path.append('/Users/benjamin/Github/pg_task/Walmart_Sales_Forecast')
# sys.path.append(os.path.abspath('..'))

from src.entities import FeatureParams, LSTMParams, PathParams, TrainingPipelineParams
from src.train import train_process

from typing import NoReturn, List, Tuple

import pytest
import pandas as pd
from faker import Faker
from datetime import datetime

ROW_NUMS = 1000

@pytest.fixture(scope="session")
def synthetic_data_path() -> str:
    return "tests/synthetic_data.csv"

@pytest.fixture(scope="session")
def load_model_path() -> str:
    return "tests/test_model_"


@pytest.fixture(scope="session")
def metric_path() -> str:
    return "tests/"

@pytest.fixture(scope="package")
def path_params(synthetic_data_path: str,
                load_model_path:str,
                metric_path:str) -> PathParams:
    path_param = PathParams(
        input_data_path=synthetic_data_path,
        output_model_path=load_model_path,
        metric_path=metric_path
    )
    return path_param

@pytest.fixture(scope="session")
def synthetic_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        
        "Store": [fake.pyint(min_value=1, max_value=10) for _ in range(ROW_NUMS)],
        "Weekly_Sales": [fake.pyfloat(min_value=1000000, max_value=5000000) for _ in range(ROW_NUMS)],
        "Holiday_Flag": [fake.pyint(min_value=0, max_value=1) for _ in range(ROW_NUMS)],
        "Temperature": [fake.pyfloat(min_value=30, max_value=40) for _ in range(ROW_NUMS)],
        "Fuel_Price": [fake.pyfloat(min_value=2.1, max_value=2.3) for _ in range(ROW_NUMS)],
        "CPI": [fake.pyfloat(min_value=200, max_value=205) for _ in range(ROW_NUMS)],
        "Unemployment": [fake.pyfloat(min_value=8.1, max_value=8.3) for _ in range(ROW_NUMS)],
    }

    return pd.DataFrame(data=df)

@pytest.fixture(scope="session")
def synthetic_date_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(21)
    df = {
        "Date": [fake.date_between_dates(date_start=datetime(2015,1,1), date_end=datetime(2019,12,31)) for _ in range(ROW_NUMS)],
        "Store": [fake.pyint(min_value=1, max_value=10) for _ in range(ROW_NUMS)],
        "Weekly_Sales": [fake.pyfloat(min_value=1000000, max_value=5000000) for _ in range(ROW_NUMS)],
        "Holiday_Flag": [fake.pyint(min_value=0, max_value=1) for _ in range(ROW_NUMS)],
        "Temperature": [fake.pyfloat(min_value=30, max_value=40) for _ in range(ROW_NUMS)],
        "Fuel_Price": [fake.pyfloat(min_value=2.1, max_value=2.3) for _ in range(ROW_NUMS)],
        "CPI": [fake.pyfloat(min_value=200, max_value=205) for _ in range(ROW_NUMS)],
        "Unemployment": [fake.pyfloat(min_value=8.1, max_value=8.3) for _ in range(ROW_NUMS)],
    }

    return pd.DataFrame(data=df)

@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment"
    ]


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return [
        "Holiday_Flag"
    ]


@pytest.fixture(scope="session")
def target_col() -> str:
    return "Weekly_Sales"

@pytest.fixture(scope="session")
def datetime_col() -> str:
    return "Date"

@pytest.fixture(scope="package")
def feature_params(categorical_features: List[str],
                   numerical_features: List[str],
                   target_col: str,
                   datetime_col:str) -> FeatureParams:
    features = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        datetime_col=datetime_col
    )
    return features

@pytest.fixture(scope="package")
def lstm_params() -> LSTMParams:
    model = LSTMParams(
        time_step=3,
        lstm_units=50,
        dense_unit=1,
        model_type="LSTM",
        optimizer='adam',
        loss='mean_squared_error',
        train_rate=0.8,
        dropout=0.1,
        recurrent_dropout=0.1,
        layer_norm_axis=1,
        beta_initializer='ones',
        gamma_initializer='zeros',
        learning_rate=0.0015,
        epochs=2,
        batch_size=1
    )
    return model



@pytest.fixture(scope="package")
def train_pipeline_params(
    path_params: PathParams,
    lstm_params: LSTMParams,
    feature_params: FeatureParams
) -> TrainingPipelineParams:

    train_pipeline_parms = TrainingPipelineParams(
        path_config=path_params,
        model_params=lstm_params,
        feature_params=feature_params
    )
    return train_pipeline_parms

# @pytest.fixture(scope="package")
# def train_synthetic(train_pipeline_params: TrainingPipelineParams) -> NoReturn:
#     train_process(train_pipeline_params)