"""Feature engineering module for heart attack prediction model."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from ..config import FEATURE_CONFIG

def create_interaction_features(df, features=None):
    """Create interaction features between specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to create interactions for
        
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    if features is None:
        features = FEATURE_CONFIG['interaction_features']
    
    df_copy = df.copy()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feat1, feat2 = features[i], features[j]
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_{feat2}_interaction"
                df_copy[interaction_name] = df[feat1] * df[feat2]
    
    return df_copy

def create_polynomial_features(df, features=None, degree=None):
    """Create polynomial and interaction features for specified numerical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of features to create polynomials and interactions for
        degree (int): Degree of polynomial features
        
    Returns:
        pd.DataFrame: DataFrame with polynomial and interaction features
    """
    if features is None:
        features = FEATURE_CONFIG['interaction_features']
    if degree is None:
        degree = FEATURE_CONFIG['polynomial_degree']
    
    df_copy = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    if not features:
        return df_copy
        
    feature_data = df[features]
    poly_features = poly.fit_transform(feature_data)
    
    # Get feature names and clean them
    feature_names = poly.get_feature_names_out(features)
    
    # Add polynomial features to dataframe, with cleaned names
    for i, name in enumerate(feature_names[len(features):], len(features)):
        # Replace spaces with underscores and add suffix for clarity
        clean_name = name.replace(' ', '_') + '_poly'
        df_copy[clean_name] = poly_features[:, i]
    
    return df_copy

def engineer_features(df):
    """Apply all feature engineering steps.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    df = create_interaction_features(df)
    df = create_polynomial_features(df)
    
    return df 