from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# Create necessary directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# In-memory user database for demonstration
# In production, this should be replaced with a proper database
users = {
    'demo': {
        'password': generate_password_hash('energy123'),
        'email': 'demo@example.com'
    }
}

# Authentication routes
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if username in users and check_password_hash(users[username]['password'], password):
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    if username in users:
        return jsonify({'success': False, 'message': 'Username already exists'}), 409
    
    users[username] = {
        'password': generate_password_hash(password),
        'email': email
    }
    
    return jsonify({'success': True, 'message': 'User created successfully'}), 201

# Data processing routes
@app.route('/process_data', methods=['POST'])
def process_data():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Basic validation
        required_columns = ['timestamp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'success': False, 
                'message': f'Missing required columns: {", ".join(missing_columns)}'
            }), 400
        
        # Basic preprocessing
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        # Handle missing values
        df = df.dropna(subset=['timestamp'])  # Remove rows with missing timestamps
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Save processed data
        processed_file_path = 'data/processed_data.csv'
        df.to_csv(processed_file_path, index=False)
        
        # Return summary statistics
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(numeric_columns)
        }
        
        return jsonify({
            'success': True,
            'message': 'Data processed successfully',
            'stats': stats
        }), 200
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'}), 500

# Model training and prediction routes
@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    data = request.json
    model_type = data.get('model_type', 'isolation_forest')
    threshold = float(data.get('threshold', 0.5))
    
    try:
        # Load processed data
        df = pd.read_csv('data/processed_data.csv')
        
        # Extract features for model
        features = df.select_dtypes(include=[np.number]).columns
        X = df[features].values
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        anomalies = None
        anomaly_scores = None
        
        # Apply selected model
        if model_type == 'isolation_forest':
            model = IsolationForest(contamination=threshold, random_state=42)
            anomalies = model.fit_predict(X_scaled)
            # Convert to binary labels where -1 is anomaly, 1 is normal
            anomalies = np.where(anomalies == -1, 1, 0)
            anomaly_scores = model.decision_function(X_scaled)
            # Invert scores so higher values mean more anomalous
            anomaly_scores = -anomaly_scores
            
        elif model_type == 'kmeans':
            model = KMeans(n_clusters=2, random_state=42)
            clusters = model.fit_predict(X_scaled)
            distances = np.min(np.sqrt(np.sum((X_scaled - model.cluster_centers_[clusters.reshape(-1, 1)])**2, axis=2)), axis=1)
            # Normalize distances
            anomaly_scores = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
            # Apply threshold
            anomalies = np.where(anomaly_scores > threshold, 1, 0)
            
        elif model_type == 'autoencoder':
            # Define the autoencoder model
            input_dim = X_scaled.shape[1]
            encoding_dim = max(int(input_dim / 2), 1)
            
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation='relu')(input_layer)
            decoder = Dense(input_dim, activation='sigmoid')(encoder)
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train the autoencoder
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, 
                           validation_split=0.2, callbacks=[early_stopping], verbose=0)
            
            # Get reconstruction error
            predictions = autoencoder.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - predictions, 2), axis=1)
            
            # Normalize scores
            anomaly_scores = (mse - np.min(mse)) / (np.max(mse) - np.min(mse))
            # Apply threshold
            anomalies = np.where(anomaly_scores > threshold, 1, 0)
        
        # Add results to dataframe
        df['anomaly'] = anomalies
        df['anomaly_score'] = anomaly_scores
        
        # Calculate summary statistics
        anomaly_count = df['anomaly'].sum()
        normal_count = len(df) - anomaly_count
        anomaly_percentage = (anomaly_count / len(df)) * 100
        
        # Save results
        results_path = 'data/anomaly_results.csv'
        df.to_csv(results_path, index=False)
        
        # Create time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Time-based anomaly distributions
            hour_anomalies = df.groupby('hour')['anomaly'].sum().to_dict()
            day_anomalies = df.groupby('day_of_week')['anomaly'].sum().to_dict()
            month_anomalies = df.groupby('month')['anomaly'].sum().to_dict()
        else:
            hour_anomalies = {}
            day_anomalies = {}
            month_anomalies = {}
        
        # Feature importance for isolation forest
        feature_importance = {}
        if model_type == 'isolation_forest':
            importances = model.feature_importances_
            for i, feature in enumerate(features):
                feature_importance[feature] = float(importances[i])
        
        return jsonify({
            'success': True,
            'message': 'Anomaly detection completed',
            'stats': {
                'anomaly_count': int(anomaly_count),
                'normal_count': int(normal_count),
                'anomaly_percentage': float(anomaly_percentage),
                'hour_anomalies': hour_anomalies,
                'day_anomalies': day_anomalies,
                'month_anomalies': month_anomalies,
                'feature_importance': feature_importance,
                'model_type': model_type,
                'threshold': threshold
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error detecting anomalies: {str(e)}'
        }), 500

# Get results route
@app.route('/get_results', methods=['GET'])
def get_results():
    try:
        if not os.path.exists('data/anomaly_results.csv'):
            return jsonify({
                'success': False,
                'message': 'No results found. Run detection first.'
            }), 404
            
        df = pd.read_csv('data/anomaly_results.csv')
        
        # Convert to JSON
        results = df.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving results: {str(e)}'
        }), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
