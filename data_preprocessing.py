"""
Data Preprocessing Module for Credit Scoring ML Project
Handles data loading, cleaning, encoding, normalization, and class balancing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


def load_data(filepath='credit_risk_dataset.csv'):
    """
    Load the credit risk dataset from CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    """
    Handle missing values using mean/median imputation
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values imputed
    """
    print("\nHandling missing values...")
    print(f"Missing values before imputation:\n{df.isnull().sum()}")
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from imputation if present
    if 'loan_status' in numerical_cols:
        numerical_cols.remove('loan_status')
    
    # Impute numerical columns with median
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Impute categorical columns with most frequent value
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    print(f"Missing values after imputation:\n{df.isnull().sum().sum()} total")
    return df


def prepare_data(filepath='credit_risk_dataset.csv', test_size=0.2, random_state=42):
    """
    Main function to orchestrate the entire preprocessing pipeline
    
    Args:
        filepath: Path to the CSV file
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test (preprocessed and ready for modeling)
    """
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Separate features and target
    print("\nSeparating features and target...")
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Class imbalance ratio: {y.value_counts()[1] / y.value_counts()[0]:.2f}")
    
    # Identify feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing pipeline
    print("\nApplying preprocessing transformations...")
    
    # Define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ])
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE for class balancing
    print("\nApplying SMOTE for class balancing...")
    print(f"Before SMOTE - Train set class distribution:\n{pd.Series(y_train).value_counts()}")
    
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    print(f"After SMOTE - Train set class distribution:\n{pd.Series(y_train_balanced).value_counts()}")
    
    print("\n" + "="*60)
    print("Data preprocessing completed successfully!")
    print("="*60)
    
    return X_train_balanced, X_test_processed, y_train_balanced, y_test


if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
