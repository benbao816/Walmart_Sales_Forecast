from src.models.build_model import build_lstm
from src.entities.lstm_params import LSTMParams
from src.features import create_dataset, parse_cate_cols, scale_data

import pandas as pd
from keras.models import Sequential
from typing import NoReturn, List, Tuple
from sklearn.utils.validation import check_is_fitted

def test_build_lstm(lstm_params:str,
                    feature_params:str,
                    synthetic_data: pd.DataFrame):
    
    test_sales_data_normalized, _ = scale_data(synthetic_data, feature_params)
    time_step = lstm_params.time_step
    X, y, shape = create_dataset(test_sales_data_normalized, time_step)
    test_model = build_lstm(lstm_param=lstm_params,
                            features=X,
                            target=y,
                            shape=shape)
    # assert type(test_model) == Sequential
    assert 'loss' in [k for k, v in test_model.history.history.items()]
    assert isinstance(test_model, Sequential)