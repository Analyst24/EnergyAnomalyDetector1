import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import offline utilities
try:
    from utils.offline import log_offline_activity, DATA_DIR, CACHE_DIR
except ImportError:
    # Fallback if offline module is not available
    def log_offline_activity(activity_type, details=None):
        pass
    DATA_DIR = "data"
    CACHE_DIR = "cache"
    
    # Ensure directories exist
    for directory in [DATA_DIR, CACHE_DIR]:
        os.makedirs(directory, exist_ok=True)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_data(file_path):
    """
    Load data from a CSV file with offline support
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with loaded data
    """
    try:
        # Log the data loading attempt
        log_offline_activity("data_load_attempt", {"file_path": file_path})
        
        # Check if file exists
        if not os.path.exists(file_path):
            # If the path doesn't include the data directory, try looking there
            if DATA_DIR not in file_path:
                alternative_path = os.path.join(DATA_DIR, os.path.basename(file_path))
                if os.path.exists(alternative_path):
                    file_path = alternative_path
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Cache a copy in the data directory for offline use
        cache_file_path = os.path.join(DATA_DIR, f"cached_{os.path.basename(file_path)}")
        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
        df.to_csv(cache_file_path, index=False)
        
        # Log success
        log_offline_activity("data_load_success", {
            "file_path": file_path,
            "cached_path": cache_file_path,
            "rows": len(df),
            "columns": len(df.columns)
        })
        
        return df
    except Exception as e:
        # Log the error
        log_offline_activity("data_load_error", {"file_path": file_path, "error": str(e)})
        
        # Try to load from cache as fallback
        try:
            cache_file_path = os.path.join(DATA_DIR, f"cached_{os.path.basename(file_path)}")
            if os.path.exists(cache_file_path):
                df = pd.read_csv(cache_file_path)
                log_offline_activity("data_load_from_cache", {
                    "file_path": cache_file_path,
                    "rows": len(df),
                    "columns": len(df.columns)
                })
                return df
        except Exception:
            pass
        
        # If all else fails, raise the original error
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """
    Preprocess the input dataframe with offline support
    
    Args:
        df: pandas DataFrame with raw data
        
    Returns:
        Preprocessed pandas DataFrame
    """
    try:
        # Log start of preprocessing
        log_offline_activity("preprocessing_start", {
            "original_shape": df.shape,
            "columns": list(df.columns)
        })
        
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
        
        # Track missing value counts before handling
        missing_before = df_processed.isnull().sum().to_dict()
        
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
        
        # Track missing value counts after handling
        missing_after = df_processed.isnull().sum().to_dict()
        
        # Cache the processed dataframe for offline use
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_path = os.path.join(CACHE_DIR, f"processed_data_{timestamp}.csv")
            df_processed.to_csv(cache_path, index=False)
        except Exception as e:
            # If caching fails, just log and continue
            log_offline_activity("preprocessing_cache_error", {"error": str(e)})
        
        # Log preprocessing completion
        log_offline_activity("preprocessing_complete", {
            "processed_shape": df_processed.shape,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "added_columns": list(set(df_processed.columns) - set(df.columns))
        })
        
        return df_processed
    
    except Exception as e:
        # Log error
        log_offline_activity("preprocessing_error", {"error": str(e)})
        # Re-raise the exception
        raise

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
