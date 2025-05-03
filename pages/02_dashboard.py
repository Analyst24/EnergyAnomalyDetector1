import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from utils.auth import require_auth
from utils.visualizations import (
    create_energy_overview,
    create_anomaly_distribution,
    create_hourly_analysis,
    create_feature_correlation
)

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Dashboard - Energy Anomaly Detection",
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
        default_index=1,
    )
    
    # Handle navigation
    if selected != "Dashboard":
        if selected == "Home":
            st.switch_page("pages/01_home.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            pages_list = ['upload_data', 'run_detection', 'results', 'model_insights', 'recommendations', 'settings', 'logout']
            page_index = pages_list.index(page_name)
            st.switch_page(f"pages/0{page_index + 3}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Energy Dashboard</h1>", unsafe_allow_html=True)

# Check if data exists
has_data = st.session_state.get('current_data') is not None
has_results = st.session_state.get('detection_results') is not None

if not has_data:
    # Display demo data if no actual data
    st.info("No data uploaded yet. Showing sample dashboard visualizations.")
    
    # Create sample data for demonstration
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=1000, freq='H')
    
    # Create consumption with daily and weekly patterns
    hour_effect = np.sin(np.pi * dates.hour / 24) * 10
    day_effect = np.sin(np.pi * dates.dayofweek / 7) * 5
    
    consumption = 100 + hour_effect + day_effect
    
    # Add trend and noise
    consumption += np.linspace(0, 10, len(dates))
    consumption += np.random.normal(0, 5, len(dates))
    
    # Create anomalies
    anomaly_indices = np.random.choice(range(len(dates)), size=50, replace=False)
    anomalies = np.zeros(len(dates))
    anomaly_scores = np.random.normal(0.1, 0.05, len(dates))
    
    consumption_array = consumption.values.copy()  # Convert to numpy array
    for idx in anomaly_indices:
        consumption_array[idx] *= np.random.choice([0.5, 1.5])  # Either too high or too low
        anomalies[idx] = 1
        anomaly_scores[idx] = np.random.uniform(0.6, 0.9)
    consumption = pd.Series(consumption_array, index=dates)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'consumption': consumption,
        'anomaly': anomalies,
        'anomaly_score': anomaly_scores,
        'temperature': 20 + 10 * np.sin(np.pi * np.arange(len(dates)) / (24 * 30)) + np.random.normal(0, 2, len(dates)),
        'humidity': 50 + 20 * np.sin(np.pi * np.arange(len(dates)) / (24 * 15)) + np.random.normal(0, 5, len(dates)),
        'meter_id': np.random.choice(['M001', 'M002', 'M003', 'M004', 'M005'], len(dates)),
        'location': np.random.choice(['Office', 'Factory', 'Warehouse', 'Data Center'], len(dates))
    })
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] < 18)).astype(int)
    
else:
    # Use actual data
    df = st.session_state.current_data
    
    # Check if we have results
    if has_results:
        # Merge results with data
        result_df = st.session_state.detection_results
        if 'anomaly' in result_df.columns and 'anomaly_score' in result_df.columns:
            df['anomaly'] = result_df['anomaly']
            df['anomaly_score'] = result_df['anomaly_score']
    else:
        st.warning("Data loaded but no anomaly detection results found. Run detection to see anomalies.")

# Create dashboard with multiple visualizations
st.markdown("### Energy Consumption Overview")

# Energy consumption overview
fig1 = create_energy_overview(df)
st.plotly_chart(fig1, use_container_width=True)

# Create 3 columns for summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    if 'anomaly' in df.columns:
        anomaly_count = df['anomaly'].sum()
        anomaly_pct = (anomaly_count / len(df)) * 100
        st.metric("Anomalies Detected", f"{anomaly_count}", f"{anomaly_pct:.1f}%")
    else:
        st.metric("Anomalies Detected", "N/A")

with col2:
    if 'consumption' in df.columns:
        avg_consumption = df['consumption'].mean()
        max_consumption = df['consumption'].max()
        st.metric("Average Consumption", f"{avg_consumption:.1f}", f"Max: {max_consumption:.1f}")
    else:
        st.metric("Average Consumption", "N/A")

with col3:
    if 'anomaly' in df.columns and 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        peak_hour = df.groupby('hour')['anomaly'].sum().idxmax()
        st.metric("Peak Anomaly Hour", f"{peak_hour}:00")
    else:
        st.metric("Peak Anomaly Hour", "N/A")

# Create 2 columns for charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Anomaly Distribution")
    if 'anomaly' in df.columns:
        fig2 = create_anomaly_distribution(df)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No anomaly data available. Run detection first.")

with col2:
    st.markdown("### Time of Day Analysis")
    if 'anomaly' in df.columns and 'timestamp' in df.columns:
        fig3 = create_hourly_analysis(df)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No time-based anomaly data available.")

# Feature correlation
st.markdown("### Feature Correlation")
fig4 = create_feature_correlation(df)
st.plotly_chart(fig4, use_container_width=True)

# Location-based anomaly distribution
if 'location' in df.columns and 'anomaly' in df.columns:
    st.markdown("### Location-based Anomaly Distribution")
    
    # Group by location and calculate anomaly statistics
    location_stats = df.groupby('location').agg({
        'anomaly': ['sum', 'mean'],
        'consumption': 'mean'
    }).reset_index()
    
    location_stats.columns = ['location', 'anomaly_count', 'anomaly_rate', 'avg_consumption']
    location_stats['anomaly_rate'] = location_stats['anomaly_rate'] * 100
    
    # Create bar chart
    fig5 = px.bar(
        location_stats,
        x='location',
        y='anomaly_count',
        color='avg_consumption',
        labels={
            'location': 'Location',
            'anomaly_count': 'Number of Anomalies',
            'avg_consumption': 'Avg. Consumption'
        },
        title='Anomalies by Location',
        color_continuous_scale='Viridis',
        template='plotly_dark'
    )
    
    # Add text labels
    fig5.update_traces(
        text=location_stats['anomaly_rate'].round(1).astype(str) + '%',
        textposition='outside'
    )
    
    fig5.update_layout(
        xaxis_title="Location",
        yaxis_title="Number of Anomalies",
        coloraxis_colorbar_title="Avg. Consumption"
    )
    
    st.plotly_chart(fig5, use_container_width=True)

# Actions section
st.markdown("### Actions")
action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("Upload New Data", use_container_width=True):
        st.switch_page("pages/03_upload_data.py")

with action_col2:
    if st.button("Run Detection", use_container_width=True):
        st.switch_page("pages/04_run_detection.py")

with action_col3:
    if st.button("View Detailed Results", use_container_width=True):
        st.switch_page("pages/05_results.py")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
