from typing import Union
from dataclasses import dataclass

from .feature_params import FeatureParams
from .lstm_params import  LSTMParams
from .path_params import PathParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    path_config: PathParams
    model_params: LSTMParams
    feature_params: FeatureParams
    # model_params:ModelParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
