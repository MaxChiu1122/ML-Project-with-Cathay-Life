"""Heart Attack Prediction Project package."""

from .config import DATA_PATH, MODEL_CONFIG, FEATURE_CONFIG, TRAIN_CONFIG
from .feature_engineering.feature_engineering import (
    create_polynomial_features,
    engineer_features
)
from .model import HeartAttackPredictor

__all__ = [
    'DATA_PATH',
    'MODEL_CONFIG',
    'FEATURE_CONFIG',
    'TRAIN_CONFIG',
    'create_polynomial_features',
    'engineer_features',
    'HeartAttackPredictor'
]
