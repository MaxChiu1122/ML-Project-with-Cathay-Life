import pandas as pd

def clean_data(input_path, output_path, target_column="HadHeartAttack"):
    """
    Clean the input dataset by handling missing values and renaming the target column.
    Save the cleaned data to a new CSV file.

    Parameters:
    - input_path (str): Path to the raw input CSV file (e.g., sample_data.csv).
    - output_path (str): Path to save the cleaned CSV file (e.g., cleaned_sample_data.csv).
    - target_column (str): The new name for the target column (default "HadHeartAttack").
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv(input_path)
    print("Columns before processing:", data.columns.tolist())
    
    # Handle missing values
    print("Handling missing values...")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in numerical_cols:
        if col != target_column:
            data[col] = data[col].fillna(data[col].median())
    for col in categorical_cols:
        if col != target_column:
            data[col] = data[col].fillna(data[col].mode()[0])
    
    # Drop rows where target_column is NaN
    print(f"Dropping rows where '{target_column}' is NaN...")
    before_drop = len(data)
    data = data.dropna(subset=[target_column])
    after_drop = len(data)
    print(f"Dropped {before_drop - after_drop} rows.")

    print(f"Missing values after processing:\n{data.isnull().sum()}")
    
    # Save cleaned data
    print(f"Saving cleaned data to {output_path}...")
    data.to_csv(output_path, index=False)
    print("Cleaned data saved successfully.")
    print("Columns after processing:", data.columns.tolist())



# if __name__ == "__main__":
#     input_path = '../data/sample_data.csv'
#     output_path = '../data/cleaned_sample_data.csv'
#     clean_data(input_path, output_path)