import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.entities.feature_params import FeatureParams


def one_hot_encode(*arrays):
    return [pd.get_dummies(arr).values for arr in arrays]

def inverse_y_array(y, shape, scaler):
    dummies = [0] * (shape - 1)
    y_reform = np.array([[i] + dummies if type(i) != np.ndarray else [i[0]] + dummies for i in y ])
    rescale_y = scaler.inverse_transform(y_reform)
    rescale_y = [round(i[0],2) for i in rescale_y]
    return rescale_y

def create_dataset(dataset, time_steps=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), :]
        data_x.append(a)
        data_y.append(dataset[i + time_steps, 0]) 
    return np.array(data_x), np.array(data_y), np.array(data_x).shape

def parse_cate_cols(df:pd.DataFrame, params:FeatureParams):
    cate_encoded = [one_hot_encode(df[cate])[0] for cate in params.categorical_features]
    stack_cols = [df[params.target_col], df[params.numerical_features]] + cate_encoded
    sales_with_features = np.column_stack(stack_cols)
    return sales_with_features

def scale_data(df:pd.DataFrame, params:FeatureParams):
    scaler = MinMaxScaler(feature_range=(0, 1))
    # sales_with_features = parse_cate_cols(df, params)
    sales_data_normalized = scaler.fit_transform(df)
    return sales_data_normalized, scaler

def build_feature(df:pd.DataFrame, params:FeatureParams) -> pd.DataFrame:
    df_copy = df[[params.datetime_col,params.target_col]].copy()
    df_copy[params.datetime_col] = pd.to_datetime(df_copy[params.datetime_col], dayfirst=True)
    time_sales = df_copy.groupby(params.datetime_col).sum().reset_index()

    exclude_col = [params.datetime_col, params.target_col] + [params.categorical_features]
    feature_to_process = [i for i in df.columns.tolist() if i not in exclude_col] 
    feature_date_col = [params.datetime_col] + feature_to_process
    func_list = ['mean', 'min', 'max']
    col_rename_dict = dict()
    for func in func_list:
        new_col = [i + '_' + func for i in feature_to_process]
        col_rename = dict(zip(feature_to_process,new_col))
        col_rename_dict[func] = col_rename

    data_list = [time_sales.drop(params.datetime_col, axis = 1),
                df[feature_date_col].groupby(params.datetime_col).mean()
                                    .rename(columns=col_rename_dict['mean'])
                                    .reset_index()
                                    .drop(params.datetime_col, axis = 1),
                df[feature_date_col].groupby(params.datetime_col).min()
                                    .rename(columns=col_rename_dict['min'])
                                    .reset_index()
                                    .drop(params.datetime_col, axis = 1),
                df[feature_date_col].groupby(params.datetime_col).max()
                                    .rename(columns=col_rename_dict['max'])
                                    .reset_index()
                                    .drop(params.datetime_col, axis = 1),
                df[[params.datetime_col] + params.categorical_features].drop_duplicates()
                                               .drop(params.datetime_col, axis = 1)
                ]
    feature = pd.concat(data_list, axis = 1)
    return feature



    
    
    

