import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import h2o
from h2o.automl import H2OAutoML
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage by converting data types.
    """
    # Convert object columns to category
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        df[col] = df[col].astype('category')
    
    # Convert integer columns to smallest possible type
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        max_val = df[col].max()
        min_val = df[col].min()
        
        if min_val >= 0:
            if max_val <= 255:
                df[col] = df[col].astype(np.uint8)
            elif max_val <= 65535:
                df[col] = df[col].astype(np.uint16)
        else:
            if min_val >= -128 and max_val <= 127:
                df[col] = df[col].astype(np.int8)
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = df[col].astype(np.int16)
    
    return df

def load_and_preprocess_data(
    file_path: str,
    target_col: str,
    feature_selection: bool = True,
    correlation_threshold: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and preprocess data with optimized memory usage and optional feature selection.
    """
    # Read data with optimized chunk size
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Feature selection if enabled
    selected_features = list(df.columns)
    if feature_selection and target_col in df.columns:
        # Remove highly correlated features
        correlation_matrix = df.corr().abs()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        
        # Make sure we don't drop the target column
        if target_col in to_drop:
            to_drop.remove(target_col)
            
        df = df.drop(columns=to_drop)
        selected_features = list(df.columns)
    
    return df, selected_features

def train_h2o_model(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    max_runtime_secs: int = 3600,
    max_models: int = 20,
    seed: int = 42
) -> Tuple[H2OAutoML, dict]:
    """
    Train an H2O AutoML model with optimized settings.
    """
    # Initialize H2O
    h2o.init()
    
    # Convert to H2O frame
    train = h2o.H2OFrame(df)
    
    # Set up AutoML
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        seed=seed,
        sort_metric="AUC",
        verbosity="info"
    )
    
    # Train model
    aml.train(
        x=features,
        y=target_col,
        training_frame=train
    )
    
    # Get model performance metrics
    performance = {
        'auc': aml.leader.auc(),
        'logloss': aml.leader.logloss(),
        'accuracy': aml.leader.accuracy()[0][1],
        'precision': aml.leader.precision()[0][1],
        'recall': aml.leader.recall()[0][1],
        'f1': aml.leader.F1()[0][1]
    }
    
    return aml, performance

def save_model_artifacts(
    model: H2OAutoML,
    selected_features: List[str],
    performance: dict,
    output_dir: str
) -> None:
    """
    Save model artifacts and metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = str(output_path / "model")
    h2o.save_model(model.leader, model_path)
    
    # Save features list
    joblib.dump(selected_features, output_path / "selected_features.joblib")
    
    # Save performance metrics
    joblib.dump(performance, output_path / "performance_metrics.joblib") 