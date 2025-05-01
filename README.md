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
  - Class imbalance exists — techniques like resampling and class weight adjustment are needed.

## Goals
- Conduct Exploratory Data Analysis (EDA).
- Apply and compare classification models (Logistic Regression, SVM, Random Forest, etc.).
- Handle class imbalance properly.
- Identify key variables affecting heart disease likelihood.


## References
- Original dataset and description adapted from CDC BRFSS survey (https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)
- Inspiration:
  - Imbalance treatment techniques in machine learning (https://www.kaggle.com/code/zeyadusf/imbalance-treatment-techniques-in-machine-learning)
  - Insurance Prediction- LGBM,GBM,XGBoost EDA (https://www.kaggle.com/code/drfrank/insurance-prediction-lgbm-gbm-xgboost-eda)
  - Classification models for Biomechanical Features (https://www.kaggle.com/code/shahriyarmammadli/classification-models-for-biomechanical-features/notebook)
  - Seabron & Plotly for Beginners (https://www.kaggle.com/code/drfrank/seabron-plotly-for-beginners)

## 技術組架構
1. Data cleaning : 找多種方法且說明原因
2. EDA、各個特徵的分布，須了解每個特徵背後的涵義
3. Outlier 處理 : 根據不同特徵的 Outlier 可能會有不同的處理方式，邏輯較重要
4. Features enginnering : 先以 create 新的特徵為主要作法 (交互作用...)、以及對特徵做鰾準化等…
5. 建立模型 
    1. : 直接以 AutoML 為基礎去訓練，但可能就會有它會自動幫我們篩特徵的問題，怕之後會不好解釋
    2. 自己建模型去做訓練，且再利用 features importance 來去判別那些為重要變數，那些為不重要變數 (較主觀)
    3. 建立模型，最後利用 SHAP 進行特徵重要性的評估 (若是用 SHAP 的話前面 Features enginnering 的部分就不用去特別新增交互作用項，SHAP 會自動判斷)
6. 評估結果以及調參 : 結合商業問題，重點解釋為甚麼我們看中 recall，以及我們在調整參數、選特徵的邏輯