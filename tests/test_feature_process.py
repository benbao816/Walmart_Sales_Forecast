
from src.entities import FeatureParams, LSTMParams
from src.features import create_dataset, parse_cate_cols, scale_data, build_feature
import pandas as pd
from typing import NoReturn, List, Tuple


def test_parse_cate_cols(synthetic_data: pd.DataFrame,
                         feature_params:FeatureParams):
    test_sales_with_features = parse_cate_cols(synthetic_data, feature_params)
    assert len(synthetic_data) == len(test_sales_with_features)

def test_scale_data(synthetic_data: pd.DataFrame,
                    feature_params:FeatureParams):
    sales_data_normalized, _ = scale_data(synthetic_data, feature_params)
    assert len(synthetic_data) == len(sales_data_normalized)

def test_create_dataset(synthetic_data: pd.DataFrame,
                        lstm_params: LSTMParams,
                        feature_params: FeatureParams):
    sales_data_normalized, _ = scale_data(synthetic_data, feature_params)
    time_step = lstm_params.time_step
    X, y, shape = create_dataset(sales_data_normalized, time_step)
    assert shape[1] == time_step
    assert len(X) == len(y)
    assert len(X) + time_step == len(synthetic_data)

def test_build_feature(synthetic_date_data: pd.DataFrame,
                       feature_params: FeatureParams):
    feature = build_feature(synthetic_date_data, feature_params)
    isinstance(feature, pd.DataFrame)

