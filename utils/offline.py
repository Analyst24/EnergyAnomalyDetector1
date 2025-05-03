"""
Utilities for ensuring the application works in offline mode.
This module contains functions to check and handle network connectivity,
fallback to local resources, and manage offline compatibility.
"""

import os
import socket
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Ignore specific warnings that might be related to network connectivity
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*certificate verification.*")

# Constants for offline mode
OFFLINE_MODE = True  # Set to True to force the system to run in offline mode
DATA_DIR = "data"
MODELS_DIR = "models"
CACHE_DIR = "cache"

# Ensure data directories exist
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

def is_connected(host="8.8.8.8", port=53, timeout=3):
    """
    Check if internet connection is available by trying to connect to a public DNS server.
    Always returns False when OFFLINE_MODE is True.
    
    Args:
        host: Host to check connection against
        port: Port to use for connection check
        timeout: Connection timeout in seconds
        
    Returns:
        Boolean indicating whether internet connection is available
    """
    if OFFLINE_MODE:
        return False
        
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def cache_file(file_path, content, is_binary=False):
    """
    Cache content to a file for offline use
    
    Args:
        file_path: Path where to save the content
        content: Content to cache
        is_binary: Whether the content is binary
        
    Returns:
        Path to the cached file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    mode = "wb" if is_binary else "w"
    with open(file_path, mode) as f:
        f.write(content)
    
    return file_path

def get_cached_file(file_path, default=None, is_binary=False):
    """
    Retrieve content from a cached file
    
    Args:
        file_path: Path to the cached file
        default: Default value to return if file doesn't exist
        is_binary: Whether the file contains binary data
        
    Returns:
        Content of the cached file or default
    """
    if not os.path.exists(file_path):
        return default
    
    mode = "rb" if is_binary else "r"
    with open(file_path, mode) as f:
        return f.read()

def log_offline_activity(activity_type, details=None):
    """
    Log offline activity for debugging purposes
    
    Args:
        activity_type: Type of activity (e.g., 'data_load', 'model_training')
        details: Additional details about the activity
    """
    log_file = os.path.join(CACHE_DIR, "offline_log.jsonl")
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "details": details or {}
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def get_sample_data():
    """
    Get sample energy data for demo purposes when no data is available or uploaded
    
    Returns:
        Pandas DataFrame with sample energy data
    """
    # First, check if we have a sample in the data directory
    sample_path = os.path.join(DATA_DIR, "sample_energy_data.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path, parse_dates=['timestamp'])
    
    # If not, generate a synthetic sample
    start_date = pd.Timestamp("2024-01-01")
    dates = pd.date_range(start=start_date, periods=100, freq="H")
    
    np.random.seed(42)  # For reproducibility
    
    # Generate base consumption with daily and weekly patterns
    base_consumption = 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)  # Daily cycle
    base_consumption += 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / (24 * 7))  # Weekly cycle
    
    # Add random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Create anomalies (about 5%)
    anomaly_indices = np.random.choice(len(dates), size=5, replace=False)
    anomalies = np.zeros(len(dates))
    for idx in anomaly_indices:
        if np.random.random() > 0.5:
            # High anomaly
            anomalies[idx] = np.random.uniform(50, 100)
        else:
            # Low anomaly
            anomalies[idx] = np.random.uniform(-50, -20)
    
    # Combine components
    consumption = base_consumption + noise + anomalies
    
    # Create temperature data (correlated with consumption)
    temperature = 20 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.normal(0, 1, len(dates))
    
    # Create humidity data
    humidity = 50 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24 + np.pi) + np.random.normal(0, 3, len(dates))
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'consumption': consumption,
        'temperature': temperature,
        'humidity': humidity,
        'meter_id': 'METER001',
        'location': 'Building A'
    })
    
    # Add derived features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17) & ~df['is_weekend'].astype(bool)).astype(int)
    
    # Save the sample for future use
    df.to_csv(sample_path, index=False)
    
    return df

def is_package_available(package_name):
    """
    Check if a Python package is available without importing it
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        Boolean indicating whether the package is available
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Initialize offline mode on module import
if __name__ != "__main__":
    # Log the application start in offline mode
    log_offline_activity("startup", {
        "offline_mode": OFFLINE_MODE,
        "internet_available": is_connected()
    })