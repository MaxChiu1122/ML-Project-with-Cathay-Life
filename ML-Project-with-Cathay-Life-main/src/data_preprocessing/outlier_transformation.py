import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mstats, zscore, boxcox, shapiro, probplot
import matplotlib.pyplot as plt
import seaborn as sns


def winsorization(df, column, lower_percentile=0.01, upper_percentile=0.01):
    """
    Winsorization: Capping outliers at the specified percentiles.
    """
    df_transformed = df.copy(deep=True) 
    df_transformed[column] = mstats.winsorize(df_transformed[column], limits=[lower_percentile, upper_percentile])
    return df_transformed

def log_transformation(df, column):
    """
    Log Transformation: Apply log(x+1) to handle skewed distributions.
    """
    df_transformed = df.copy(deep=True) 
    df_transformed[column] = np.log1p(df_transformed[column])  # log(x+1) to handle zeros and negative values
    return df_transformed

def z_score_transformation(df, column, threshold=3):
    """
    Z-Score Transformation: Replace outliers with the threshold value.
    """
    df_transformed = df.copy(deep=True) 
    df_transformed['z_score'] = zscore(df_transformed[column])
    df_transformed[column] = np.where(df_transformed['z_score'] > threshold, threshold, 
                                      np.where(df_transformed['z_score'] < -threshold, -threshold, df_transformed[column]))
    return df_transformed

def iqr_transformation(df, column):
    """
    IQR (Interquartile Range) Transformation: Cap values outside the IQR boundaries.
    """
    df_transformed = df.copy(deep=True) 
    Q1 = df_transformed[column].quantile(0.25)
    Q3 = df_transformed[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_transformed[column] = np.where(df_transformed[column] < lower_bound, lower_bound, 
                                      np.where(df_transformed[column] > upper_bound, upper_bound, df_transformed[column]))
    return df_transformed

def boxcox_transformation(df, column):
    """
    Box-Cox Transformation: Apply Box-Cox transformation to normalize the data.
    """
    df_transformed = df.copy(deep=True) 
    df_transformed[column], _ = boxcox(df_transformed[column] + 1)  # Box-Cox needs strictly positive values, so add 1 if necessary
    return df_transformed

def transform_outliers(df, column, method='winsorization', **kwargs):
    """
    Main function to choose the outlier handling method.
    """
    if method == 'winsorization':
        return winsorization(df, column, **kwargs)
    elif method == 'log':
        return log_transformation(df, column)
    elif method == 'z_score':
        return z_score_transformation(df, column, **kwargs)
    elif method == 'iqr':
        return iqr_transformation(df, column)
    elif method == 'boxcox':
        return boxcox_transformation(df, column)
    else:
        raise ValueError(f"Method '{method}' is not supported!")
    
def plot_transformed(df, column, transformed_df, transformed_column, method):
    """
    Plot the original and transformed data as separate images.
    """
    plt.figure(figsize=(14, 6))

    # Plot Original Data (Before Transformation)
    plt.subplot(1, 2, 1)
    sns.kdeplot(df[column], fill=True, color='blue', alpha=0.5, linewidth=2)
    plt.title(f'Original {column} - Distribution')
    plt.xlabel(column)
    plt.ylabel('Density')

    # Boxplot of Original Data
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column], palette='Set2')
    plt.title(f'Original {column} - Boxplot')

    plt.tight_layout()
    plt.show()

    # Plot Transformed Data (After Transformation)
    plt.figure(figsize=(14, 6))

    # Distribution (KDE) of Transformed Data
    plt.subplot(1, 2, 1)
    sns.kdeplot(transformed_df[transformed_column], fill=True, color='red', alpha=0.5, linewidth=2)
    plt.title(f'{method} - Transformed {column} - Distribution')
    plt.xlabel(column)
    plt.ylabel('Density')

    # Boxplot of Transformed Data
    plt.subplot(1, 2, 2)
    sns.boxplot(x=transformed_df[transformed_column], palette='Set2')
    plt.title(f'{method} - Transformed {column} - Boxplot')

    plt.tight_layout()
    plt.show()

# Shapiro-Wilk test for normality
def check_normality(df, column, method):
    stat, p_value = shapiro(df[column])
    print(f"Shapiro-Wilk Test ({method}): Statistic = {stat}, p-value = {p_value}")
    if p_value > 0.05:
        print(f"{method} transformation: The data is likely normally distributed.")
    else:
        print(f"{method} transformation: The data is likely not normally distributed.")

# QQ plot to visually inspect normality
def qq_plot(df, column, method, palette='Set2'):

    # Create the QQ plot
    plt.figure(figsize=(12, 12))
    
    # Generate the QQ plot (this will return values for quantiles and theoretical quantiles)
    res = stats.probplot(df[column], dist="norm", plot=plt)
    
    # Extract the points (x, y) from the QQ plot for further customization
    line = res[0]
    
    plt.scatter(line[0], line[1], color= '#66c2a5', label=f'{method} Transformation')

    # Set plot title and labels
    plt.title(f"QQ Plot of {column} ({method} Transformation)")
    plt.legend()

    plt.show()
