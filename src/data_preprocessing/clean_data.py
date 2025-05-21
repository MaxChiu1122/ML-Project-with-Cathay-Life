import pandas as pd
from typing import List, Optional


class DataCleaner:
    """
    A class to perform group-based imputation and end-to-end data cleaning.
    """

    def __init__(self, group_cols: Optional[List[str]] = None):
        self.group_cols = group_cols or ['SmokerStatus', 'RaceEthnicityCategory', 'AgeCategory', 'Sex']

    def clean_data(
        self,
        input_path: str,
        output_path: str,
        target_column: str = "HadHeartAttack",
        missing_row_threshold: float = 0.3
    ) -> None:
        """
        Load, clean, and save the dataset using group-based imputation.

        Parameters:
        - input_path (str): Path to the raw input CSV file.
        - output_path (str): Path to save the cleaned CSV file.
        - target_column (str): The name of the target column.
        - missing_row_threshold (float): Maximum allowed fraction of missing values per row.
        """
        print("🔹 Loading data...")
        df = pd.read_csv(input_path)
        print(f"Initial shape: {df.shape}")

        print(f"\n🔹 Dropping rows with missing target '{target_column}'...")
        before = df.shape[0]
        df = df.dropna(subset=[target_column])
        after = df.shape[0]
        print(f"Dropped {before - after} rows. Remaining: {after}")

        print(f"\n🔹 Dropping rows with more than {int(missing_row_threshold * 100)}% missing values...")
        missing_ratio = df.isnull().mean(axis=1)
        rows_before = df.shape[0]
        df = df[missing_ratio <= missing_row_threshold]
        rows_after = df.shape[0]
        print(f"Dropped {rows_before - rows_after} rows. Remaining: {rows_after}")

        print("\n🔹 Starting group-based imputation...")
        df_cleaned = self.group_based_imputation(df)

        print(f"\n🔹 Saving cleaned data to '{output_path}'...")
        df_cleaned.to_csv(output_path, index=False)
        print("✅ Cleaned data saved.")

    def group_based_imputation(self, df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        group_cols = group_cols or self.group_cols
        df = df.copy()

        print("Step 1️⃣: Imputing missing values in grouping columns...")
        df = self._handle_grouping_columns(df, group_cols)

        print("Step 2️⃣: Imputing missing values in remaining columns...")
        df = self._handle_other_columns(df, group_cols)

        print("✅ Group-based imputation completed.")
        print("🔎 Remaining missing values per column:\n", df.isnull().sum())
        return df

    def _handle_grouping_columns(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        for col in group_cols:
            if df[col].isnull().sum() == 0:
                continue

            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                mode_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(mode_val)
                print(f"Filled missing values in '{col}' with mode: {mode_val}")
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Filled missing values in '{col}' with median: {median_val}")
        return df

    def _handle_other_columns(self, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        for i, col in enumerate(df.columns):
            if col in group_cols or df[col].isnull().sum() == 0:
                continue

            print(f"Imputing column: '{col}' ({i+1}/{len(df.columns)})")

            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                df[col] = df.groupby(group_cols)[col].transform(
                    lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)

                if df[col].isnull().sum() > 0:
                    fallback = df[col].mode()
                    if not fallback.empty:
                        df[col] = df[col].fillna(fallback.iloc[0])
                        print(f"Fallback: filled remaining nulls in '{col}' with overall mode.")
            else:
                df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
                if df[col].isnull().sum() > 0:
                    fallback = df[col].median()
                    df[col] = df[col].fillna(fallback)
                    print(f"Fallback: filled remaining nulls in '{col}' with overall median.")
        return df


def clean_data(input_path: str, output_path: str, target_column: str = "HadHeartAttack", missing_row_threshold: float = 0.3) -> None:
    """
    Wrapper function to clean data using DataCleaner class.
    """
    cleaner = DataCleaner()
    cleaner.clean_data(
        input_path=input_path,
        output_path=output_path,
        target_column=target_column,
        missing_row_threshold=missing_row_threshold
    )
