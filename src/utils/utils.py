import json
import pickle
import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import NoReturn
from statsmodels.tsa.stattools import adfuller
from keras.models import load_model

from src.entities import FeatureParams

def save_metrics_to_json(file_path: str, metrics: dict) -> NoReturn:
    with open(file_path, "w") as metric_file:
        json.dump(metrics, metric_file, indent=4)


def save_pkl_file(input_file, output_name: str) -> NoReturn:
    with open(output_name, "wb") as f:
        pickle.dump(input_file, f)


def load_pkl_file(input_file: str):
    with open(input_file, "rb") as fin:
        res = pickle.load(fin)
    return res

def load_h5_file(input_file:str):
    model = load_model(input_file)
    return model

def load_latest_h5(input_path:str):
    model_path = glob.glob(f"{input_path}model_*.h5")
    if len(model_path) > 0:
        model_time = [os.path.getmtime(path) for path in model_path]
        model_files = list(zip(model_path, model_time))
        latest_model_file = [i[0] for i in model_files if i[1]==max(model_time)][0]
    else: 
        print('No .h5 files found under the directory')
        latest_model_file = None
    try:
        model = load_h5_file(latest_model_file)
    except Exception as e:
        print(str(e))

    return model

def stationarity_check(df: pd.DataFrame,
                       feature_params:FeatureParams) -> bool:
    
    result = adfuller(df[feature_params.target_col])
    p_value = result[1]

    if p_value <= 0.05:
        stationarity  = True
    else:
        stationarity  = False
    return stationarity