
# import sys
# sys.path.append('/Users/benjamin/Github/pg_task/Walmart_Sales_Forecast')

from src.train import train_process
import numpy as np
import glob
import os

def test_train_process(train_pipeline_params:str):
    test_metric_dict = train_process(train_pipeline_params)

    for k, v in test_metric_dict.items():
        isinstance(test_metric_dict[k], (int, float, np.float64))
    
    for ele in glob.glob('tests/metrics*.json'):
        assert os.path.isfile(ele)

    for ele in glob.glob('tests/model*.pkl'):
        assert os.path.isfile(ele)
        