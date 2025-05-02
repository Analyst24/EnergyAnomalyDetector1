import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from utils.auth import require_auth
from utils.data_processor import scale_features
from utils.models import detect_anomalies, train_isolation_forest, train_kmeans, train_autoencoder

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Run Detection - Energy Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create sidebar
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "Dashboard", "Upload Data", "Run Detection", "Results", 
         "Model Insights", "Recommendations", "Settings", "Logout"],
        icons=['house', 'graph-up', 'cloud-upload', 'play-circle', 'clipboard-data', 
               'tools', 'lightbulb', 'gear', 'box-arrow-right'],
        menu_icon="cast",
        default_index=3,
    )
    
    # Handle navigation
    if selected != "Run Detection":
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "upload_data", "results", "model_insights", 
                  "recommendations", "settings", "logout"].index(page_name)
            if idx < 2:
                st.switch_page(f"pages/0{idx + 2}_{page_name}.py")
            else:
                st.switch_page(f"pages/0{idx + 3}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Run Anomaly Detection</h1>", unsafe_allow_html=True)

# Check if data is available
if 'current_data' not in st.session_state or st.session_state.current_data is None:
    st.warning("No data available. Please upload data first.")
    
    if st.button("Go to Upload Data", use_container_width=True):
        st.switch_page("pages/03_upload_data.py")
else:
    # Display data summary
    df = st.session_state.current_data
    st.markdown(f"### Data Summary: {len(df)} rows, {len(df.columns)} columns")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    with col2:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    
    with col3:
        if 'timestamp' in df.columns:
            time_range = f"{df['timestamp'].min().split(' ')[0]} to {df['timestamp'].max().split(' ')[0]}"
            st.metric("Time Range", time_range)
        else:
            st.metric("Time Range", "N/A")
    
    # Model selection
    st.markdown("### Select Detection Model")
    
    model_type = st.radio(
        "Choose an anomaly detection algorithm:",
        ["Isolation Forest", "K-Means Clustering", "Autoencoder"],
        horizontal=True,
        index=0
    )
    
    # Map selection to internal model name
    model_map = {
        "Isolation Forest": "isolation_forest",
        "K-Means Clustering": "kmeans",
        "Autoencoder": "autoencoder"
    }
    
    selected_model = model_map[model_type]
    
    # Create columns for model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Parameters")
        
        # Common parameters
        threshold = st.slider(
            "Anomaly Threshold (sensitivity)",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Higher values detect more anomalies but may include false positives"
        )
        
        # Model-specific parameters
        if selected_model == "isolation_forest":
            n_estimators = st.slider(
                "Number of Trees",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )
        elif selected_model == "kmeans":
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=2,
                step=1
            )
        elif selected_model == "autoencoder":
            epochs = st.slider(
                "Training Epochs",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
        
        # Feature selection
        st.markdown("### Feature Selection")
        
        # Get numeric columns for feature selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove any timestamp columns that might have been converted to numeric
        if 'hour' in numeric_cols:
            numeric_cols.remove('hour')
        if 'day_of_week' in numeric_cols:
            numeric_cols.remove('day_of_week')
        if 'month' in numeric_cols:
            numeric_cols.remove('month')
        if 'year' in numeric_cols:
            numeric_cols.remove('year')
        
        # Selected features
        selected_features = st.multiselect(
            "Select features for anomaly detection:",
            options=numeric_cols,
            default=numeric_cols
        )
        
        if not selected_features:
            st.warning("Please select at least one feature for anomaly detection.")
    
    with col2:
        st.markdown("### Model Description")
        
        if selected_model == "isolation_forest":
            st.markdown("""
            **Isolation Forest** works by isolating observations by randomly selecting a feature and a split value. 
            It's particularly effective at detecting outliers in high-dimensional datasets.
            
            **Best for:**
            - Datasets with multiple features
            - When anomalies are clearly different from normal data
            - Quick detection with minimal tuning
            
            **Parameters:**
            - **Threshold**: Controls the proportion of points to be classified as anomalies
            - **Number of Trees**: More trees provide better accuracy but slower processing
            """)
        elif selected_model == "kmeans":
            st.markdown("""
            **K-Means Clustering** groups similar data points together. Anomalies are detected as points 
            that are far from their nearest cluster center.
            
            **Best for:**
            - Datasets where normal data forms natural clusters
            - When anomalies are distant from normal clusters
            - Visualizing different groups in your data
            
            **Parameters:**
            - **Threshold**: Controls the sensitivity of anomaly detection
            - **Number of Clusters**: Number of groups to divide the data into
            """)
        elif selected_model == "autoencoder":
            st.markdown("""
            **Autoencoder** is a neural network that learns to compress and reconstruct the input data.
            Anomalies are points with high reconstruction error.
            
            **Best for:**
            - Complex, non-linear patterns in data
            - Capturing subtle deviations from normal behavior
            - Datasets with many features and complex relationships
            
            **Parameters:**
            - **Threshold**: Controls the sensitivity of anomaly detection
            - **Training Epochs**: Number of training iterations (more epochs mean better learning but take longer)
            """)
    
    # Run detection button
    if st.button("Run Anomaly Detection", use_container_width=True, type="primary", disabled=not selected_features):
        if not selected_features:
            st.error("Please select at least one feature for anomaly detection.")
        else:
            try:
                with st.spinner(f"Running {model_type} anomaly detection..."):
                    # Prepare the data
                    X = df[selected_features].copy()
                    
                    # Handle missing values
                    X = X.fillna(X.mean())
                    
                    # Scale the features
                    X_scaled, scaler = scale_features(X, scaler_type='standard')
                    
                    # Run the selected model
                    start_time = time.time()
                    
                    if selected_model == "isolation_forest":
                        model, anomalies, scores = train_isolation_forest(
                            X_scaled.values, 
                            contamination=threshold
                        )
                    elif selected_model == "kmeans":
                        model, anomalies, scores = train_kmeans(
                            X_scaled.values, 
                            n_clusters=n_clusters, 
                            contamination=threshold
                        )
                    elif selected_model == "autoencoder":
                        model, anomalies, scores = train_autoencoder(
                            X_scaled.values, 
                            contamination=threshold
                        )
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Add results to dataframe
                    result_df = df.copy()
                    result_df['anomaly'] = anomalies
                    result_df['anomaly_score'] = scores
                    
                    # Calculate summary statistics
                    anomaly_count = int(np.sum(anomalies))
                    normal_count = len(anomalies) - anomaly_count
                    anomaly_percentage = (anomaly_count / len(anomalies)) * 100
                    
                    # Store results in session state
                    st.session_state.detection_results = result_df
                    
                    # Store detection metadata
                    st.session_state.detection_metadata = {
                        'model_type': selected_model,
                        'threshold': threshold,
                        'anomaly_count': anomaly_count,
                        'normal_count': normal_count,
                        'anomaly_percentage': anomaly_percentage,
                        'processing_time': processing_time,
                        'selected_features': selected_features
                    }
                    
                    # Add time-based statistics if timestamp is available
                    if 'timestamp' in result_df.columns:
                        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
                        result_df['hour'] = result_df['timestamp'].dt.hour
                        result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek
                        
                        # Hour-based anomalies
                        hour_anomalies = result_df.groupby('hour')['anomaly'].sum().to_dict()
                        day_anomalies = result_df.groupby('day_of_week')['anomaly'].sum().to_dict()
                        
                        st.session_state.detection_metadata['hour_anomalies'] = hour_anomalies
                        st.session_state.detection_metadata['day_anomalies'] = day_anomalies
                    
                    # Add feature importance for Isolation Forest
                    if selected_model == 'isolation_forest':
                        feature_importance = dict(zip(selected_features, model.feature_importances_))
                        st.session_state.detection_metadata['feature_importance'] = feature_importance
                
                # Show success message
                st.success(f"Anomaly detection completed in {processing_time:.2f} seconds!")
                
                # Show summary statistics
                st.markdown("### Detection Results")
                
                # Create metrics row
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Anomalies Detected", anomaly_count)
                
                with metric_cols[1]:
                    st.metric("Normal Points", normal_count)
                
                with metric_cols[2]:
                    st.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")
                
                with metric_cols[3]:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Show preview of results
                st.markdown("### Results Preview")
                st.dataframe(
                    result_df[['anomaly', 'anomaly_score'] + selected_features].head(10),
                    use_container_width=True
                )
                
                # Visualization of anomalies
                st.markdown("### Anomaly Score Distribution")
                
                # Create histogram of anomaly scores
                fig = px.histogram(
                    result_df, 
                    x='anomaly_score',
                    color='anomaly',
                    color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
                    marginal='box',
                    title='Distribution of Anomaly Scores',
                    template='plotly_dark'
                )
                
                # Add threshold line
                fig.add_vline(
                    x=np.percentile(result_df['anomaly_score'], 100 * (1 - threshold)),
                    line_dash='dash',
                    line_color='white',
                    annotation_text=f'Threshold: {threshold}'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Call to action buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("View Detailed Results", use_container_width=True):
                        st.switch_page("pages/05_results.py")
                
                with col2:
                    if st.button("See Model Insights", use_container_width=True):
                        st.switch_page("pages/06_model_insights.py")
                
            except Exception as e:
                st.error(f"Error during anomaly detection: {str(e)}")
    
    # Show previous results if available
    elif 'detection_results' in st.session_state and st.session_state.detection_results is not None:
        st.markdown("### Previous Detection Results")
        
        if 'detection_metadata' in st.session_state:
            metadata = st.session_state.detection_metadata
            
            # Create metrics row
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Anomalies Detected", metadata.get('anomaly_count', 'N/A'))
            
            with metric_cols[1]:
                st.metric("Normal Points", metadata.get('normal_count', 'N/A'))
            
            with metric_cols[2]:
                percentage = metadata.get('anomaly_percentage', 'N/A')
                if percentage != 'N/A':
                    percentage = f"{percentage:.2f}%"
                st.metric("Anomaly Percentage", percentage)
            
            with metric_cols[3]:
                model_name = metadata.get('model_type', 'unknown')
                model_display = {
                    'isolation_forest': 'Isolation Forest',
                    'kmeans': 'K-Means',
                    'autoencoder': 'Autoencoder'
                }.get(model_name, model_name)
                st.metric("Model Used", model_display)
            
            # Show call to action
            st.info("Previous detection results are available. Run a new detection or view the existing results.")
            
            # Call to action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("View Detailed Results", key="view_prev_results", use_container_width=True):
                    st.switch_page("pages/05_results.py")
            
            with col2:
                if st.button("See Model Insights", key="view_prev_insights", use_container_width=True):
                    st.switch_page("pages/06_model_insights.py")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
