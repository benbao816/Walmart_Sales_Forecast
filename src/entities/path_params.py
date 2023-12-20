from dataclasses import dataclass


@dataclass
class PathParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    
