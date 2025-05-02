import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import json
import time

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def train_isolation_forest(X, contamination=0.05):
    """
    Train an Isolation Forest model for anomaly detection
    
    Args:
        X: numpy array of shape (n_samples, n_features) with input features
        contamination: expected proportion of anomalies (between 0 and 0.5)
        
    Returns:
        Trained model and anomaly predictions (-1 for anomalies, 1 for normal)
    """
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X)
    predictions = model.predict(X)
    scores = model.decision_function(X)
    
    # In Isolation Forest, anomalies are -1, normal points are 1
    # Convert to binary where 1 is anomaly, 0 is normal
    anomalies = np.where(predictions == -1, 1, 0)
    
    # Scores are negative for anomalies, convert to positive for consistency
    anomaly_scores = -scores
    
    # Save model
    model_path = 'models/isolation_forest.joblib'
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_type': 'isolation_forest',
        'contamination': contamination,
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_count': int(np.sum(anomalies)),
        'normal_count': int(X.shape[0] - np.sum(anomalies))
    }
    
    with open('models/isolation_forest_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    return model, anomalies, anomaly_scores

def train_kmeans(X, n_clusters=2, contamination=0.05):
    """
    Train a K-Means model for anomaly detection
    
    Args:
        X: numpy array of shape (n_samples, n_features) with input features
        n_clusters: number of clusters to form
        contamination: expected proportion of anomalies
        
    Returns:
        Trained model, anomaly predictions (1 for anomalies, 0 for normal)
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    
    # Calculate distance to nearest centroid
    distances = []
    for i in range(X.shape[0]):
        point = X[i].reshape(1, -1)
        min_dist = float('inf')
        for j in range(n_clusters):
            centroid = model.cluster_centers_[j].reshape(1, -1)
            dist = np.sqrt(np.sum((point - centroid) ** 2))
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)
    
    distances = np.array(distances)
    
    # Normalize distances to 0-1 range
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    normalized_distances = (distances - min_dist) / (max_dist - min_dist)
    
    # Set threshold based on contamination level
    threshold = np.percentile(normalized_distances, 100 * (1 - contamination))
    
    # Points with distances > threshold are anomalies
    anomalies = (normalized_distances > threshold).astype(int)
    anomaly_scores = normalized_distances
    
    # Save model
    model_path = 'models/kmeans.joblib'
    joblib.dump(model, model_path)
    
    # Save threshold and other metadata
    metadata = {
        'model_type': 'kmeans',
        'n_clusters': n_clusters,
        'contamination': contamination,
        'threshold': float(threshold),
        'min_distance': float(min_dist),
        'max_distance': float(max_dist),
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_count': int(np.sum(anomalies)),
        'normal_count': int(X.shape[0] - np.sum(anomalies))
    }
    
    with open('models/kmeans_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    return model, anomalies, anomaly_scores

def train_autoencoder(X, contamination=0.05):
    """
    Train an Autoencoder for anomaly detection
    
    Args:
        X: numpy array of shape (n_samples, n_features) with input features
        contamination: expected proportion of anomalies
        
    Returns:
        Trained model, anomaly predictions (1 for anomalies, 0 for normal)
    """
    # Define architecture
    input_dim = X.shape[1]
    encoding_dim = max(int(input_dim / 2), 1)  # at least 1 neuron
    
    # Build autoencoder model
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(encoding_dim * 2, activation='relu')(input_layer)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(encoding_dim * 2, activation='relu')(encoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.0001,
        restore_best_weights=True
    )
    
    history = autoencoder.fit(
        X, X,
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Calculate reconstruction error
    predictions = autoencoder.predict(X)
    mse = np.mean(np.power(X - predictions, 2), axis=1)
    
    # Normalize scores
    normalized_mse = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))
    
    # Set threshold based on contamination level
    threshold = np.percentile(normalized_mse, 100 * (1 - contamination))
    
    # Points with MSE > threshold are anomalies
    anomalies = (normalized_mse > threshold).astype(int)
    anomaly_scores = normalized_mse
    
    # Save model
    model_path = 'models/autoencoder'
    autoencoder.save(model_path)
    
    # Save threshold and other metadata
    metadata = {
        'model_type': 'autoencoder',
        'encoding_dim': encoding_dim,
        'contamination': contamination,
        'threshold': float(threshold),
        'min_mse': float(np.min(mse)),
        'max_mse': float(np.max(mse)),
        'num_samples': X.shape[0],
        'num_features': X.shape[1],
        'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'anomaly_count': int(np.sum(anomalies)),
        'normal_count': int(X.shape[0] - np.sum(anomalies)),
        'training_epochs': len(history.history['loss']),
        'final_loss': float(history.history['loss'][-1])
    }
    
    with open('models/autoencoder_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    return autoencoder, anomalies, anomaly_scores

def detect_anomalies(X, model_type='isolation_forest', contamination=0.05):
    """
    Detect anomalies using the specified model
    
    Args:
        X: numpy array of shape (n_samples, n_features) with input features
        model_type: type of model to use ('isolation_forest', 'kmeans', or 'autoencoder')
        contamination: expected proportion of anomalies
        
    Returns:
        anomalies: binary array (1 for anomalies, 0 for normal)
        scores: anomaly scores (higher = more anomalous)
    """
    if model_type == 'isolation_forest':
        model, anomalies, scores = train_isolation_forest(X, contamination)
    elif model_type == 'kmeans':
        model, anomalies, scores = train_kmeans(X, n_clusters=2, contamination=contamination)
    elif model_type == 'autoencoder':
        model, anomalies, scores = train_autoencoder(X, contamination)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return anomalies, scores
