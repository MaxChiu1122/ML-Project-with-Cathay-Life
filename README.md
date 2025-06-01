# ML-Project-with-Cathay-Life

## Overview
This project is part of the Machine Learning and Artificial Intelligence course in collaboration with **Cathay Life**. In this project, we work on real-world problems in the financial and healthcare industries using machine learning methods.

Our task focuses on predicting heart disease based on key health indicators collected by the **CDC** through the Behavioral Risk Factor Surveillance System (BRFSS). Using machine learning models, we aim to identify important risk factors and improve prediction accuracy.

## Dataset
- **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS)
- **Data Description**:  
  - 2022 annual survey data from 400,000+ adults.
  - Dataset with 40 relevant variables linked to heart disease risk.
  - Binary target variable: `HadHeartAttack` ("Yes" or "No").
  - Class imbalance exists — techniques like resampling, class weighting, and SMOTE are used to address it.

## Project Workflow
The project workflow consists of several stages:

### 1. Data Preprocessing and Cleaning
- **Data Loading and Exploration**: We load the raw data and explore its structure, checking for missing values and performing initial visualizations.
- **Data Cleaning**: Missing values are handled using a group-based imputation strategy. Rows with more than 30% missing data are excluded, and imputation is performed based on the median or mode within defined groups.
- **Feature Engineering**: Features are encoded, and some new features are created using interaction terms and binning techniques to categorize continuous variables like BMI and Sleep Hours.
- **Outlier Detection and Transformation**: Local and global outlier detection methods, such as Isolation Forest (IForest) and SMOTENC, are applied to clean the data further.

### 2. Feature Engineering
- **Encoding**: Categorical features are encoded using LabelEncoder and OneHotEncoder.
- **Scaling**: Non-binary numerical features are standardized using `StandardScaler`.
- **Interaction Features**: Interaction terms are created between continuous and categorical features to capture non-linear relationships.
  
### 3. Model Selection, Hyperparameter Tuning, and Evaluation
- **Model Selection**: H2O AutoML is used to select and train the best classification models. The models are trained on the processed data using both original and SMOTE-augmented data.
- **Revenue Optimization**: Custom objective functions are defined based on revenue calculation. False Negatives (FN) are penalized more heavily than True Negatives (TN), reflecting the business context.
- **Hyperparameter Tuning**: Hyperparameter tuning is performed using `Hyperopt` for both LightGBM and XGBoost, optimizing for the revenue score instead of traditional accuracy.
- **Model Evaluation**: The best models are evaluated using various metrics, including accuracy, precision, recall, F1 score, AUC, and a custom revenue function. SHAP analysis is performed to interpret model feature importance.

### 4. Generative Adversarial Networks (GAN)
- **GANs for Data Augmentation**: GANs are trained on the minority class (patients with heart disease) to generate synthetic data and balance the dataset. This synthetic data is then combined with real data to train a final model.

### 5. Final Model and Results
- **Best Model Selection**: The best models from H2O AutoML, LightGBM, XGBoost, and GAN-augmented data are compared based on their revenue performance on the test set.
- **Results**: The model with the highest revenue gain over the baseline is selected. The confusion matrix, AUC curve, and SHAP force plots are generated for the final model.

## References
- Original dataset and description adapted from CDC BRFSS survey (https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)
- Inspiration:
  - Imbalance treatment techniques in machine learning (https://www.kaggle.com/code/zeyadusf/imbalance-treatment-techniques-in-machine-learning)
  - Insurance Prediction- LGBM,GBM,XGBoost EDA (https://www.kaggle.com/code/drfrank/insurance-prediction-lgbm-gbm-xgboost-eda)
  - Classification models for Biomechanical Features (https://www.kaggle.com/code/shahriyarmammadli/classification-models-for-biomechanical-features/notebook)
  - Seabron & Plotly for Beginners (https://www.kaggle.com/code/drfrank/seabron-plotly-for-beginners)
