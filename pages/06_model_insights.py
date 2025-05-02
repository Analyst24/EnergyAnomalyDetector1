import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from utils.auth import require_auth
from utils.visualizations import (
    create_anomaly_scores_histogram,
    create_confusion_matrix,
    create_feature_importance
)

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Model Insights - Energy Anomaly Detection",
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
        default_index=5,
    )
    
    # Handle navigation
    if selected != "Model Insights":
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "upload_data", "run_detection", "results", 
                  "recommendations", "settings", "logout"].index(page_name)
            if idx < 4:
                st.switch_page(f"pages/0{idx + 2}_{page_name}.py")
            else:
                st.switch_page(f"pages/0{idx + 3}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Model Insights</h1>", unsafe_allow_html=True)

# Check if results are available
if 'detection_results' not in st.session_state or st.session_state.detection_results is None:
    st.warning("No detection results available. Please run anomaly detection first.")
    
    if st.button("Go to Run Detection", use_container_width=True):
        st.switch_page("pages/04_run_detection.py")
else:
    # Get results data
    result_df = st.session_state.detection_results
    
    # Check if metadata is available
    if 'detection_metadata' in st.session_state:
        metadata = st.session_state.detection_metadata
    else:
        metadata = {
            'model_type': 'unknown',
            'threshold': 0.05,
            'anomaly_count': result_df['anomaly'].sum() if 'anomaly' in result_df.columns else 0,
            'normal_count': len(result_df) - result_df['anomaly'].sum() if 'anomaly' in result_df.columns else 0,
            'anomaly_percentage': (result_df['anomaly'].sum() / len(result_df) * 100) if 'anomaly' in result_df.columns else 0
        }
    
    # Display model information
    st.markdown("### Model Information")
    
    # Get model name for display
    model_name = metadata.get('model_type', 'unknown')
    model_display = {
        'isolation_forest': 'Isolation Forest',
        'kmeans': 'K-Means Clustering',
        'autoencoder': 'Autoencoder Neural Network'
    }.get(model_name, model_name.title())
    
    # Create columns for model info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Used", model_display)
    
    with col2:
        threshold = metadata.get('threshold', 'N/A')
        if threshold != 'N/A':
            threshold = f"{threshold:.3f}"
        st.metric("Threshold", threshold)
    
    with col3:
        processing_time = metadata.get('processing_time', 'N/A')
        if processing_time != 'N/A':
            processing_time = f"{processing_time:.2f}s"
        st.metric("Processing Time", processing_time)
    
    # Model-specific information
    st.markdown("### Model Description")
    
    if model_name == 'isolation_forest':
        st.markdown("""
        **Isolation Forest** is an algorithm designed to detect anomalies by isolating observations.
        
        **How it works:**
        1. It builds an ensemble of isolation trees for the dataset
        2. Anomalies are isolated in fewer steps (shorter paths in the trees)
        3. The algorithm calculates an anomaly score based on path length
        
        **Strengths:**
        - Efficient with high-dimensional data
        - Handles irrelevant features well
        - Fast training and prediction time
        - Doesn't make assumptions about data distribution
        
        **Interpretation:**
        - Higher anomaly scores indicate more anomalous points
        - Feature importance shows which variables contribute most to anomaly detection
        """)
    elif model_name == 'kmeans':
        st.markdown("""
        **K-Means Clustering** detects anomalies by identifying points that are far from their cluster centers.
        
        **How it works:**
        1. Groups similar data points into clusters
        2. Calculates the distance from each point to its nearest cluster center
        3. Points with the highest distances are flagged as anomalies
        
        **Strengths:**
        - Intuitive and easy to interpret
        - Works well when normal data forms natural clusters
        - Efficient for large datasets
        
        **Interpretation:**
        - Higher anomaly scores indicate greater distance from normal clusters
        - Location-based clusters may reveal patterns in anomaly distribution
        """)
    elif model_name == 'autoencoder':
        st.markdown("""
        **Autoencoder Neural Network** learns to compress and reconstruct normal data patterns.
        
        **How it works:**
        1. The neural network learns to compress (encode) and then reconstruct (decode) normal patterns
        2. Calculates reconstruction error for each data point
        3. Points with high reconstruction error are anomalies
        
        **Strengths:**
        - Can capture complex, non-linear patterns
        - Adaptive to various data types
        - Excellent at detecting subtle anomalies
        
        **Interpretation:**
        - Higher anomaly scores indicate greater reconstruction error
        - Works like a learned compression - anomalies are harder to compress and reconstruct
        """)
    
    # Model performance
    st.markdown("### Anomaly Score Distribution")
    
    # Create histogram of anomaly scores
    fig1 = create_anomaly_scores_histogram(result_df)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Feature importance if available
    if 'feature_importance' in metadata:
        st.markdown("### Feature Importance")
        
        # Get feature importance
        importance = metadata['feature_importance']
        features = list(importance.keys())
        values = list(importance.values())
        
        # Create feature importance visualization
        fig2 = create_feature_importance(features, values)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Feature importance explanation
        st.markdown("""
        **Feature Importance Interpretation:**
        - Higher values indicate features that are more important for anomaly detection
        - Features with higher importance have greater impact on the model's decisions
        - Consider focusing on these features when investigating anomalies
        """)
    
    # Create confusion matrix using pseudo-labels
    st.markdown("### Detection Performance")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Detection Metrics", "Advanced Analysis"])
    
    with tab1:
        # Selected features
        if 'selected_features' in metadata:
            st.markdown("#### Features Used for Detection")
            st.write(", ".join(metadata['selected_features']))
        
        # Sensitivity analysis
        st.markdown("#### Threshold Sensitivity")
        
        # Simulate different thresholds
        threshold_values = [0.01, 0.02, 0.05, 0.1, 0.15]
        anomaly_rates = []
        
        for thresh in threshold_values:
            # For simplicity, we'll use a percentile-based threshold on anomaly scores
            if 'anomaly_score' in result_df.columns:
                cutoff = np.percentile(result_df['anomaly_score'], 100 * (1 - thresh))
                anomaly_count = (result_df['anomaly_score'] > cutoff).sum()
                anomaly_rates.append(anomaly_count / len(result_df) * 100)
            else:
                anomaly_rates.append(0)
        
        # Create line chart
        threshold_df = pd.DataFrame({
            'Threshold': threshold_values,
            'Anomaly Rate (%)': anomaly_rates
        })
        
        fig3 = px.line(
            threshold_df, x='Threshold', y='Anomaly Rate (%)',
            markers=True,
            title='Anomaly Rate vs. Threshold',
            template='plotly_dark'
        )
        
        # Add current threshold marker
        current_threshold = metadata.get('threshold', 0.05)
        current_anomaly_rate = metadata.get('anomaly_percentage', 0)
        
        fig3.add_trace(
            go.Scatter(
                x=[current_threshold],
                y=[current_anomaly_rate],
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Current Threshold'
            )
        )
        
        fig3.update_layout(
            xaxis_title='Threshold Value',
            yaxis_title='Anomaly Rate (%)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Explanation
        st.markdown("""
        **Threshold Sensitivity:**
        - Lower threshold values detect more anomalies but may include false positives
        - Higher threshold values are more conservative, detecting only the most extreme anomalies
        - The optimal threshold depends on your specific use case and the cost of false positives vs. false negatives
        """)
    
    with tab2:
        # For demonstration purposes, create a simulated confusion matrix
        # In a real scenario, we would need labeled data to evaluate the model
        
        # Create a 2D visualization of data points if we have at least 2 features
        if 'selected_features' in metadata and len(metadata['selected_features']) >= 2:
            st.markdown("#### 2D Data Visualization")
            
            # Get the top 2 important features or first 2 selected features
            if 'feature_importance' in metadata:
                importance = metadata['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:2]
                top_feature_names = [f[0] for f in top_features]
            else:
                top_feature_names = metadata['selected_features'][:2]
            
            # Create scatter plot
            if all(feature in result_df.columns for feature in top_feature_names):
                fig4 = px.scatter(
                    result_df, 
                    x=top_feature_names[0], 
                    y=top_feature_names[1],
                    color='anomaly',
                    color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
                    title=f'Anomaly Detection in 2D Feature Space',
                    labels={
                        top_feature_names[0]: top_feature_names[0],
                        top_feature_names[1]: top_feature_names[1],
                        'anomaly': 'Anomaly'
                    },
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig4, use_container_width=True)
                
                # Explanation
                st.markdown(f"""
                **Feature Space Visualization:**
                - Normal points (green) and anomalies (red) plotted in 2D feature space using {top_feature_names[0]} and {top_feature_names[1]}
                - Clusters of normal points indicate common patterns in the data
                - Isolated points are more likely to be detected as anomalies
                - The model uses all selected features, not just these two dimensions
                """)
            else:
                st.info(f"Cannot create 2D visualization - features {top_feature_names} not available in the dataset.")
        
        # Time-based detection patterns if timestamp is available
        if 'timestamp' in result_df.columns and 'anomaly' in result_df.columns:
            st.markdown("#### Temporal Detection Patterns")
            
            # Make sure timestamp is datetime
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            
            # Resample to daily anomaly counts
            result_df['date'] = result_df['timestamp'].dt.date
            daily_anomalies = result_df.groupby('date')['anomaly'].sum().reset_index()
            daily_counts = result_df.groupby('date')['anomaly'].count().reset_index()
            daily_anomalies['count'] = daily_counts['anomaly']
            daily_anomalies['rate'] = daily_anomalies['anomaly'] / daily_anomalies['count'] * 100
            
            # Create time series plot
            fig5 = go.Figure()
            
            # Add bar chart for anomaly counts
            fig5.add_trace(
                go.Bar(
                    x=daily_anomalies['date'],
                    y=daily_anomalies['anomaly'],
                    name='Anomaly Count',
                    marker_color='#E74C3C'
                )
            )
            
            # Add line chart for anomaly rate
            fig5.add_trace(
                go.Scatter(
                    x=daily_anomalies['date'],
                    y=daily_anomalies['rate'],
                    name='Anomaly Rate (%)',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color='#F39C12', width=2)
                )
            )
            
            # Update layout
            fig5.update_layout(
                title='Daily Anomaly Detection Patterns',
                xaxis_title='Date',
                yaxis=dict(title='Anomaly Count'),
                yaxis2=dict(
                    title='Anomaly Rate (%)',
                    overlaying='y',
                    side='right',
                    range=[0, max(daily_anomalies['rate']) * 1.2]
                ),
                legend=dict(orientation='h', y=1.1),
                template='plotly_dark',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig5, use_container_width=True)
    
    # Model comparison (if multiple models have been run)
    st.markdown("### Model Comparison")
    
    # For now, we'll show a simulated comparison
    # In a real app, you would store results from multiple model runs
    
    st.info("Run different models on the same dataset to compare their performance.")
    
    # Create a sample comparison table
    comparison_data = {
        'Model': ['Isolation Forest', 'K-Means', 'Autoencoder'],
        'Anomaly Rate (%)': [5.0, 4.2, 6.1],
        'Processing Time (s)': [0.8, 0.5, 4.2],
        'Strengths': [
            'Good with high-dimensional data',
            'Fast and intuitive clustering',
            'Captures complex patterns'
        ]
    }
    
    # Highlight the current model
    current_model_map = {
        'isolation_forest': 'Isolation Forest',
        'kmeans': 'K-Means',
        'autoencoder': 'Autoencoder'
    }
    current_model = current_model_map.get(model_name, model_name)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create styled dataframe
    st.dataframe(
        comparison_df.style.apply(
            lambda x: ['background-color: #1E3D59' if v == current_model else '' for v in x], 
            axis=0
        ),
        use_container_width=True
    )
    
    # Actions section
    st.markdown("### Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run New Detection", use_container_width=True):
            st.switch_page("pages/04_run_detection.py")
    
    with col2:
        if st.button("View Results", use_container_width=True):
            st.switch_page("pages/05_results.py")
    
    with col3:
        if st.button("View Recommendations", use_container_width=True):
            st.switch_page("pages/07_recommendations.py")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
