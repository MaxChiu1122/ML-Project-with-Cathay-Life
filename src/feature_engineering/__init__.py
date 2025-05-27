"""Feature engineering package for heart attack prediction."""

from .feature_engineering import (
    create_interaction_features,
    create_polynomial_features,
    engineer_features
)
from ..data_preprocessing.encoding import encode_features

__all__ = [
    'create_interaction_features',
    'create_polynomial_features',
    'engineer_features',
    'encode_features'
] 