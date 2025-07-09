import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Data loading function
def load_data():
    """Load all required datasets"""
    print("Loading datasets...")
    
    # Main datasets
    train_data = pd.read_parquet('train_data.parquet')
    test_data = pd.read_parquet('test_data.parquet')
    
    # Additional datasets
    add_trans = pd.read_parquet('add_trans.parquet')
    add_event = pd.read_parquet('add_event.parquet')
    offer_metadata = pd.read_parquet('offer_metadata.parquet')
    
    # Data dictionary
    data_dict = pd.read_csv('data_dictionary.csv')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Transaction data shape: {add_trans.shape}")
    print(f"Event data shape: {add_event.shape}")
    print(f"Offer metadata shape: {offer_metadata.shape}")
    
    return train_data, test_data, add_trans, add_event, offer_metadata, data_dict

# Initial data exploration
def explore_data(train_data, test_data):
    """Perform initial data exploration"""
    print("\n=== DATA EXPLORATION ===")
    
    # Basic info
    print("\nTrain Data Info:")
    print(f"Shape: {train_data.shape}")
    print(f"Columns: {train_data.columns.tolist()[:10]}...")  # First 10 columns
    
    # Check for target variable (click)
    if 'click' in train_data.columns:
        print(f"\nTarget variable distribution:")
        print(train_data['click'].value_counts())
        print(f"Click rate: {train_data['click'].mean():.4f}")
    
    # Missing values
    print(f"\nMissing values in train: {train_data.isnull().sum().sum()}")
    print(f"Missing values in test: {test_data.isnull().sum().sum()}")
    
    # Data types
    print(f"\nData types:")
    print(train_data.dtypes.value_counts())
    
    return train_data, test_data

# Usage
if __name__ == "__main__":
    # Load data
    train_data, test_data, add_trans, add_event, offer_metadata, data_dict = load_data()
    
    # Explore data
    train_data, test_data = explore_data(train_data, test_data)