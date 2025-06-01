# ML-Project-with-Cathay-Life

## Overview
This project, in collaboration with **Cathay Life**, focuses on predicting heart disease using machine learning. The data is sourced from the **CDC**'s Behavioral Risk Factor Surveillance System (BRFSS), and our goal is to identify key risk factors and improve prediction accuracy.

## Dataset
- **Source**: CDC BRFSS 2022 survey data
- **Key Variables**: 40+ variables linked to heart disease risk
- **Target Variable**: `HadHeartAttack` ("Yes" or "No")
- **Challenges**: Class imbalance, addressed through resampling, class weighting, and SMOTE

## Workflow
### 1. Data Preprocessing
- **Data Loading & Exploration**: Examine structure and missing values
- **Cleaning**: Impute missing values using group-based imputation (median/mode), excluding rows with >30% missing data
- **Feature Engineering**: Encode features, create interaction terms, and bin continuous variables (e.g., BMI, Sleep Hours)
- **Outlier Handling**: Apply Isolation Forest and SMOTENC for outlier detection

### 2. Feature Engineering
- **Encoding**: LabelEncode and OneHotEncode categorical features
- **Scaling**: Standardize numerical features (except binary ones)
- **Interaction Terms**: Create non-linear interaction features between continuous and categorical variables

### 3. Model Training & Tuning
- **Model Selection**: Use H2O AutoML, LightGBM, XGBoost, with SMOTE and GAN-augmented data
- **Custom Objective**: Define revenue-based objective, penalizing False Negatives (FN) more than True Negatives (TN)
- **Hyperparameter Tuning**: Perform hyperparameter optimization using `Hyperopt`
- **Data Augmentation**: Use GANs to generate synthetic data for the minority class

### 4. Evaluation & Results
- **Best Model**: Compare models from H2O AutoML, LightGBM, SMOTE, and GAN-augmented data based on revenue performance
- **Evaluation Metrics**: Use accuracy, precision, recall, F1 score, AUC, and SHAP analysis for feature importance
- **Final Model**: Select the model with the highest revenue gain, and generate confusion matrix, AUC curve, and SHAP plots

## References
- CDC BRFSS dataset: [Link to Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)
- Additional Resources:
  - [Imbalance treatment techniques in ML](https://www.kaggle.com/code/zeyadusf/imbalance-treatment-techniques-in-machine-learning)
  - [Insurance Prediction- LGBM, GBM, XGBoost EDA](https://www.kaggle.com/code/drfrank/insurance-prediction-lgbm-gbm-xgboost-eda)
  - [Classification models for Biomechanical Features](https://www.kaggle.com/code/shahriyarmammadli/classification-models-for-biomechanical-features/notebook)
  - [Seaborn & Plotly for Beginners](https://www.kaggle.com/code/drfrank/seabron-plotly-for-beginners)
