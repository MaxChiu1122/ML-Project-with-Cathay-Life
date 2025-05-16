"""Model training and evaluation module for heart attack prediction."""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .config import MODEL_CONFIG, TRAIN_CONFIG

class HeartAttackPredictor:
    """Class for training and evaluating heart attack prediction models."""
    
    def __init__(self):
        """Initialize H2O and class variables."""
        h2o.init()
        self.model = None
        self.feature_importance = None
    
    def prepare_data(self, df, target='HadHeartAttack'):
        """Prepare data for H2O AutoML.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target (str): Target column name
            
        Returns:
            tuple: Training and test H2O frames
        """
        # Convert to H2O frame
        h2o_df = h2o.H2OFrame(df)
        h2o_df[target] = h2o_df[target].asfactor()
        
        # Split features and target
        self.features = [col for col in df.columns if col != target]
        self.target = target
        
        # Split data
        train, test = h2o_df.split_frame(
            ratios=[1 - TRAIN_CONFIG['test_size']], 
            seed=TRAIN_CONFIG['random_state']
        )
        
        return train, test
    
    def train(self, train_frame, sort_metric="f1"):
        """Train H2O AutoML model.
        
        Args:
            train_frame (H2OFrame): Training data
            sort_metric (str): Metric to sort models by
        """
        self.model = H2OAutoML(
            max_runtime_secs=MODEL_CONFIG['max_runtime_secs'],
            seed=MODEL_CONFIG['seed'],
            nfolds=MODEL_CONFIG['nfolds'],
            sort_metric=sort_metric
        )
        
        self.model.train(
            x=self.features,
            y=self.target,
            training_frame=train_frame
        )
        
        # Get feature importance if available
        try:
            self.feature_importance = self.model.leader.varimp()
        except:
            print("Feature importance not available for this model")
    
    def evaluate(self, test_frame):
        """Evaluate model performance.
        
        Args:
            test_frame (H2OFrame): Test data
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        predictions = self.model.leader.predict(test_frame)
        performance = self.model.leader.model_performance(test_frame)
        
        # Convert to pandas for sklearn metrics
        preds = predictions.as_data_frame()['predict']
        true_labels = test_frame[self.target].as_data_frame()[self.target]
        
        # Calculate metrics
        report = classification_report(true_labels, preds, output_dict=True)
        cm = confusion_matrix(true_labels, preds)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'h2o_performance': performance
        }
    
    def plot_confusion_matrix(self, confusion_matrix):
        """Plot confusion matrix.
        
        Args:
            confusion_matrix (np.array): Confusion matrix from evaluate()
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=30):
        """Plot feature importance.
        
        Args:
            top_n (int): Number of top features to plot
        """
        if self.feature_importance is None:
            print("Feature importance not available")
            return
            
        # Convert to pandas DataFrame
        importance_df = pd.DataFrame(self.feature_importance)
        importance_df = importance_df.sort_values(by="relative_importance", ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=importance_df['variable'].head(top_n),
            x=importance_df['relative_importance'].head(top_n),
            palette='Set2'
        )
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show() 