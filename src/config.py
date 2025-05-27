"""Configuration settings for the heart attack prediction model."""

# Data paths
DATA_PATH = {
    'raw_data': 'data/raw/heart_2022.csv',
    'cleaned_data': 'data/cleaned/heart_2022_cleaned.csv',
    'engineered_data': 'data/processed/heart_2022_engineered.csv',
    'model_output': 'models/'
}

# Feature engineering settings
FEATURE_CONFIG = {
    'polynomial_degree': 2,
    'interaction_features': [
        'BMI',
        'WeightInKilograms',
        'HeightInMeters',
        'PhysicalHealthDays',
        'MentalHealthDays',
        'SleepHours',
    ]
}

# Model parameters
MODEL_CONFIG = {
    'max_runtime_secs': 1200,  # Increased from 600
    'seed': 42,
    'nfolds': 5,  # For cross-validation
}

# Training settings
TRAIN_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'class_weights': 'balanced'  # Handle class imbalance
} 