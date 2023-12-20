import sys
sys.path.append('/Users/benjamin/Github/pg_task/Walmart_Sales_Forecast')

from src.entities.train_model_params import TrainingPipelineParams, TrainingPipelineParamsSchema
from src.models.build_model import build_lstm
from src.features import create_dataset, scale_data, build_feature
from src.utils import save_pkl_file, save_metrics_to_json


import os
import hydra
import logging
import pandas as pd
import numpy as np
from omegaconf import DictConfig,OmegaConf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

train_logger = logging.getLogger("train_pipeline")


def train_process(training_pipeline_params: TrainingPipelineParams):
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    df = pd.read_csv(training_pipeline_params.path_config.input_data_path)
    train_logger.info(f'Data file is loaded from {training_pipeline_params.path_config.input_data_path}')

    train_logger.info('Features are processing...')
    features = build_feature(df, training_pipeline_params.feature_params)
    print(features.columns)
    sales_data_normalized, scaler = scale_data(features, training_pipeline_params.feature_params)

    train_logger.info('Training data is preparing...')
    time_steps = training_pipeline_params.model_params.time_step
    X, y, shape = create_dataset(sales_data_normalized, time_steps)
    train_size = int(training_pipeline_params.model_params.train_rate * len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=False)

    train_logger.info('Model training...')
    model = build_lstm(training_pipeline_params.model_params, X_train, y_train, shape)
    train_logger.info('Model is ready')

    y_predict = model.predict(X_test)

    # y_test_reform = inverse_y_array(y_test, shape[2], scaler)
    # y_predict_reform = inverse_y_array(y_predict, shape[2], scaler)

    metric_dict = {
       "model_eval": model.evaluate(X_test,y_test),
       "mae": mean_absolute_error(y_test, y_predict),
       "mse": mean_squared_error(y_test, y_predict),
       "rmse": np.sqrt(mean_squared_error(y_test, y_predict)),
    }

    metric_file = f'metrics_{time_stamp}.json'
    metric_path = f'{training_pipeline_params.path_config.metric_path}{metric_file}'
    train_logger.info('Metrics on test data is ready')
    train_logger.info(f'{metric_dict}')
    save_metrics_to_json(metric_path, metric_dict)
    train_logger.info('Metrics are stored')


    if training_pipeline_params.model_params.model_type.lower() in ['lstm']:
        model_output = training_pipeline_params.path_config.output_model_path + \
                    training_pipeline_params.model_params.model_type + \
                    '_' + time_stamp + '.h5'    
        model.save(model_output)
    else:
        model_output = training_pipeline_params.path_config.output_model_path + \
               training_pipeline_params.model_params.model_type + \
               '_' + time_stamp + '.pkl'
        save_pkl_file(model, model_output)
        
    train_logger.info(f'Model is stored as {model_output}')
    train_logger.info(f'All of Paratmeters are stored as \n{training_pipeline_params}')

    return metric_dict

@hydra.main(config_path="../configs", config_name="train_configs")
def train_pipeline_start(cfg: DictConfig):
    os.chdir(hydra.utils.to_absolute_path("."))
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    print(train_process(params))
    

if __name__ == "__main__":
    train_pipeline_start()
    print(''.join(['='] * 10 + ['Finished'] + ['='] * 10))
    
