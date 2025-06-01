# ML-Project-with-Cathay-Life

## Overview
This project was completed as part of a university-industry collaboration with **Cathay Life**. Our objective was to build a machine learning model that predicts the risk of **heart attacks** in policyholders using health-related data from the **CDC BRFSS 2022 survey**. The ultimate goal is to integrate AI into **value-based insurance (VBI)** and support proactive risk management within the insurance ecosystem.

## Business Context
- **Industry Need**: Heart disease is a major claim source, pressuring insurance profitability.
- **Solution Strategy**: Predict → Intervene → Reward — enabling early risk detection and targeted health engagement.
- **Application**:
  - Underwriting risk assessment
  - Premium discount recommendations
  - Real-time monitoring via wearable data (e.g., Fitbit, Apple Watch)
  - Integration into Cathay’s digital wellness programs

## Dataset
- **Source**: CDC BRFSS 2022 (Kaggle)
- **Samples**: Over 400,000 adults
- **Target Variable**: `HadHeartAttack` (binary)
- **Challenge**: Severe class imbalance (~5% positive rate)

## Project Pipeline

### 1. Data Preprocessing
- **Group-based imputation**: Clean missing values based on grouped statistics
- **Outlier detection**: Use Isolation Forest to detect multivariate anomalies
- **EDA**: Identified key behavioral risk factors (e.g., smoking, sleep patterns, checkup frequency)

### 2. Feature Engineering
- **Encoding**: Binary/ordinal encoding based on feature semantics
- **Binning**: Transformed continuous variables like BMI and SleepHours for better interpretability
- **Interaction Features**: Created cross-variable interaction terms (e.g., BMI × Physical Activity)

### 3. Model Training & Optimization
- **Modeling Approaches**:
  - `H2O AutoML` for rapid ensemble model selection
  - `LightGBM` + `HyperOpt` for custom tuning
  - `SMOTENC` and `GAN` to address class imbalance
- **Custom Revenue-Based Objective**:
  ```python
  Revenue = 271,139 × TN − 401,832 × FN
  ```
  Misclassifying true heart attack cases (FN) incurs a large business cost, so class weighting and threshold optimization are applied accordingly.

### 4. Evaluation & Interpretation
- **Performance Metrics**:
  - Accuracy ≈ 94.6%
  - Precision ≈ 52%
  - Recall ≈ 33%
  - AUC ≈ 0.885
- **Model Explanation**:
  - SHAP summary & force plots
  - Global feature contributions
  - Key risk factors identified: Smoking, Stroke, Angina, Sleep irregularity

## Business Application
- **Prototype Use Case**:
  - Predict high-risk clients during underwriting
  - Recommend intervention plans (e.g., lifestyle coaching)
  - Adjust premium discounts accordingly
- **Insurance Savings Example**:
  - A smoker who quits may save **~6,050 NTD/year**
  - Improved sleep and physical activity linked to lower claim risk

## Conclusion
This project showcases how machine learning can be embedded into the **insurance product cycle** — from risk identification to dynamic pricing and engagement. With further real-world integration (e.g., health app syncing, policy feedback loops), Cathay Life can pioneer a data-driven, preventive insurance model in Taiwan.

## Team Members
- 邱士展（金融四）- MaxChiu1122  
- 易可倫（金融四）- HelloAustinYi  
- 鄭達嶸（金融四）- Rod-Zheng 
- 黃以穠（風管四）- Columbia0728
- 周紹璞（經濟四）- pacificshoulder

## References
- [CDC BRFSS Dataset (Kaggle)](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)
- [Cathay Holdings Press Release](https://www.cathayholdings.com/holdings/lastest_news/news_archive/newsarticle?newsID=ZEmNh7TgxE2NKcOstDwtoA)
- SMOTE / GAN for Class Imbalance:
  - [Zeyad Usf's ML imbalance notebook](https://www.kaggle.com/code/zeyadusf/imbalance-treatment-techniques-in-machine-learning)
- EDA and SHAP References:
  - [Seaborn & Plotly Visuals](https://www.kaggle.com/code/drfrank/seabron-plotly-for-beginners)

