import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(file_path):
    """
    Load data from a CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with loaded data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """
    Preprocess the input dataframe
    
    Args:
        df: pandas DataFrame with raw data
        
    Returns:
        Preprocessed pandas DataFrame
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
        
        # Extract time-based features
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        df_processed['month'] = df_processed['timestamp'].dt.month
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        
        # Business hours flag (8 AM to 6 PM)
        df_processed['is_business_hours'] = ((df_processed['hour'] >= 8) & 
                                             (df_processed['hour'] < 18)).astype(int)
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    # For numeric columns, fill missing values with mean
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # For categorical columns, fill with mode
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    return df_processed

def scale_features(df, scaler_type='standard'):
    """
    Scale numeric features in the dataframe
    
    Args:
        df: pandas DataFrame with features
        scaler_type: Type of scaling ('standard' or 'minmax')
        
    Returns:
        DataFrame with scaled features and scaler object
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Choose scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Scale the data
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled, scaler

def check_data_quality(df):
    """
    Check data quality and identify potential issues
    
    Args:
        df: pandas DataFrame to check
        
    Returns:
        Dict with data quality metrics
    """
    quality_report = {
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'duplicate_percentage': float(df.duplicated().sum() / len(df) * 100),
        'row_count': len(df),
        'column_count': len(df.columns),
    }
    
    # Check for potential outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = {
            'count': int(outlier_count),
            'percentage': float(outlier_count / len(df) * 100)
        }
    
    quality_report['potential_outliers'] = outliers
    
    # Check for inconsistencies in energy data
    inconsistencies = []
    
    # Example: Check for zero energy consumption during business hours in non-residential locations
    if all(col in df.columns for col in ['consumption', 'hour', 'location']):
        business_hours = (df['hour'] >= 8) & (df['hour'] < 18)
        non_residential = ~df['location'].str.contains('residential', case=False, na=False)
        zero_consumption = df['consumption'] == 0
        inconsistent = business_hours & non_residential & zero_consumption
        
        if inconsistent.any():
            inconsistencies.append({
                'type': 'Zero consumption during business hours in non-residential location',
                'count': int(inconsistent.sum()),
                'rows': df[inconsistent].index.tolist()
            })
    
    quality_report['inconsistencies'] = inconsistencies
    
    return quality_report

def get_data_summary(df):
    """
    Generate a summary of the dataset
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Dict with data summary
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'numeric_columns': list(numeric_cols),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'stats': {}
    }
    
    # Summary statistics for numeric columns
    for col in numeric_cols:
        summary['stats'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std())
        }
    
    return summary
