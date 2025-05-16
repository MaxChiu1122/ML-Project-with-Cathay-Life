import pandas as pd
import numpy as np

def encode_features(df):
    # Binary Yes/No encoding
    binary_cols = [
        'PhysicalActivities', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
        'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
        'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
        'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
        'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
        'ChestScan', 'HighRiskLastYear'
    ]
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Ordinal encoding
    sex_map = {
        'Female': 0, 'Male': 1
    }
    
    general_health_map = {
        'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4
    }
    checkup_map = {
        '5 or more years ago': 0,
        'Within past 5 years (2 years but less than 5 years ago)': 1,
        'Within past 2 years (1 year but less than 2 years ago)': 2,
        'Within past year (anytime less than 12 months ago)': 3
    }
    teeth_map = {
        'All': 3, '6 or more, but not all': 2, '1 to 5': 1, 'None of them': 0
    }
    age_map = {
        f'Age {i} to {i+4}': idx + 1 for idx, i in enumerate(range(25, 80, 5))
    }
    age_map['Age 18 to 24'] = 0
    age_map['Age 80 or older'] = 12

    df['Sex'] = df['Sex'].map(sex_map)
    df['GeneralHealth'] = df['GeneralHealth'].map(general_health_map)
    df['LastCheckupTime'] = df['LastCheckupTime'].map(checkup_map)
    df['RemovedTeeth'] = df['RemovedTeeth'].map(teeth_map)
    df['AgeCategory'] = df['AgeCategory'].map(age_map)

    # Complex mappings
    df['HadDiabetes'] = df['HadDiabetes'].map({
        'No': 0, 'No, pre-diabetes or borderline diabetes': 1,
        'Yes, but only during pregnancy (female)': 2, 'Yes': 3
    })
    
    df['CovidPos'] = df['CovidPos'].map({
        'No': 0, 'Tested positive using home test without a health professional': 1,
        'Yes': 2
    })
    
    df['TetanusLast10Tdap'] = df['TetanusLast10Tdap'].map({
        'No, did not receive any tetanus shot in the past 10 years': 0,
        'Yes, received tetanus shot but not sure what type': 1,
        'Yes, received tetanus shot, but not Tdap': 2,
        'Yes, received Tdap': 3
    })
    # Smoking status
    smoker_map = {
        'Never smoked': 0,
        'Former smoker': 1,
        'Current smoker - now smokes some days': 2,
        'Current smoker - now smokes every day': 3
    }
    ecig_map = {
        'Never used e-cigarettes in my entire life': 0,
        'Not at all (right now)': 1,
        'Use them some days': 2,
        'Use them every day': 3
    }
    df['SmokerStatus'] = df['SmokerStatus'].map(smoker_map)
    df['ECigaretteUsage'] = df['ECigaretteUsage'].map(ecig_map)

    # BMI and Sleep category
    bmi_map = {
        'Underweight': 0, 'Normal weight': 1,
        'Overweight': 2, 'Obese': 3, 'Extremly Obese': 4
    }
    sleep_map = {
        'Normal Sleep': 0, 'Short Sleep': 1, 'Long Sleep': 2,
        'Very Short Sleep': 3, 'Very Long Sleep': 4
    }
    df['BMI_Category'] = df['BMI_Category'].map(bmi_map)
    df['SleepHours_Category'] = df['SleepHours_Category'].map(sleep_map)

    return df
