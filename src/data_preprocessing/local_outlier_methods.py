
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.knn import KNN

def detect_outliers(df, method='iforest', contamination=0.005, random_state=42, n_neighbors=20, 
                    show_plots=True, boxplot_cols=None, scatterplot_cols=None):
    """
    Detect and remove outliers from a DataFrame using the selected method, with logging and visualization.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        method (str): One of 'iforest', 'lof', or 'knn'.
        contamination (float): Estimated proportion of outliers.
        random_state (int): Random seed (used in iforest).
        n_neighbors (int): Number of neighbors (used in LOF and KNN).
        show_plots (bool): Whether to show plots.
        boxplot_cols (list): Columns to show in boxplots.
        scatterplot_cols (list): Two columns to show in scatter plot.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """
    df_copy = df.copy()

    # Select features based on method
    if method == 'iforest':
        features = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()
    elif method in ['lof', 'knn']:
        features = df_copy.select_dtypes(include=['float64']).columns.tolist()
    else:
        raise ValueError("Invalid method. Choose from 'iforest', 'lof', or 'knn'.")

    features = [col for col in features if col != 'HadHeartAttack']
    print(f"🔹 Using {len(features)} features for method '{method}': {features}")

    print(f"🔎 Dataset shape before outlier removal: {df_copy.shape}")

    # Fit and predict
    if method == 'iforest':
        model = IsolationForest(contamination=contamination, random_state=random_state)
        preds = model.fit_predict(df_copy[features])
    elif method == 'lof':
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds = model.fit_predict(df_copy[features])
    elif method == 'knn':
        model = KNN(contamination=contamination, n_neighbors=n_neighbors)
        model.fit(df_copy[features])
        preds = model.labels_

    df_copy['outlier'] = preds
    df_copy['outlier_label'] = df_copy['outlier'].map({-1: 'Outlier', 1: 'Inlier'})

    num_outliers = (df_copy['outlier'] == -1).sum()
    print(f"🚫 Detected outliers: {num_outliers}")
    print(f"✅ Remaining samples after removal: {df_copy.shape[0] - num_outliers}")

    # Visualize
    if show_plots:
        # Boxplots
        if boxplot_cols:
            for col in boxplot_cols:
                if col in df_copy.columns:
                    plt.figure(figsize=(10, 4))
                    sns.boxplot(x='outlier_label', y=col, data=df_copy, palette='Set2')
                    plt.title(f"Boxplot of {col} by Outlier Label ({method})")
                    plt.show()

        # Scatterplot
        if scatterplot_cols and len(scatterplot_cols) == 2:
            x, y = scatterplot_cols
            if x in df_copy.columns and y in df_copy.columns:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    x=df_copy[x],
                    y=df_copy[y],
                    hue=df_copy['outlier_label'],
                    palette='Set2',
                    alpha=0.3
                )
                plt.title(f"Scatter Plot of {x} vs {y} ({method})")
                plt.show()

    filtered_df = df_copy[df_copy['outlier'] != -1].drop(columns=['outlier', 'outlier_label'])
    return filtered_df
