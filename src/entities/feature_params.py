from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
    datetime_col: Optional[str]
