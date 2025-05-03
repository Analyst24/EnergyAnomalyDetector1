import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from utils.auth import require_auth

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Home - Energy Anomaly Detection",
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
        default_index=0,
    )
    
    # Handle navigation
    if selected and selected != "Home":
        page_name = selected.lower().replace(' ', '_')
        pages_list = ['dashboard', 'upload_data', 'run_detection', 'results', 
                      'model_insights', 'recommendations', 'settings', 'logout']
        page_index = pages_list.index(page_name)
        st.switch_page(f"pages/0{page_index + 2}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Energy Anomaly Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Advanced Machine Learning for Energy Efficiency</h3>", unsafe_allow_html=True)

# Welcome message
st.markdown(f"### Welcome, {st.session_state.username}!")

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # Create animated energy usage visualization
    st.subheader("Energy Consumption Patterns")
    
    # Generate sample data for visualization
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    
    # Base consumption pattern with weekly seasonality
    consumption = 100 + 20 * np.sin(np.arange(len(dates)) * (2 * np.pi / 7))
    
    # Add trend
    consumption += np.arange(len(dates)) * 0.1
    
    # Add random noise
    consumption += np.random.normal(0, 5, len(dates))
    
    # Add a few anomalies
    anomaly_indices = [10, 25, 50, 75]
    for idx in anomaly_indices:
        consumption[idx] *= 1.5
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'consumption': consumption,
        'is_anomaly': [1 if i in anomaly_indices else 0 for i in range(len(dates))]
    })
    
    # Create and display interactive plot
    fig = px.line(df, x='date', y='consumption', title='Energy Consumption Over Time')
    
    # Add anomaly markers
    anomalies = df[df['is_anomaly'] == 1]
    fig.add_scatter(
        x=anomalies['date'],
        y=anomalies['consumption'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='circle-open'),
        name='Anomalies'
    )
    
    # Update layout for dark theme
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        xaxis_title="Date",
        yaxis_title="Energy Consumption",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    st.markdown("### Quick Statistics")
    stats_cols = st.columns(3)
    
    with stats_cols[0]:
        st.metric(label="Energy Saved", value="15.4%", delta="4.2%")
    
    with stats_cols[1]:
        st.metric(label="Anomalies Detected", value="24", delta="-5 from last week")
    
    with stats_cols[2]:
        st.metric(label="Efficiency Score", value="87/100", delta="2")

with col2:
    # System overview
    st.subheader("System Overview")
    
    # Create animated dashboard visualization
    fig = go.Figure()
    
    # Add traces for donut charts
    fig.add_trace(go.Pie(
        labels=["Normal", "Anomalies"],
        values=[90, 10],
        hole=0.6,
        domain=dict(x=[0, 0.45], y=[0.55, 1.0]),
        marker=dict(colors=['#2ECC71', '#E74C3C'])
    ))
    
    fig.add_trace(go.Pie(
        labels=["Peak Hours", "Off Hours", "Weekend"],
        values=[50, 30, 20],
        hole=0.6,
        domain=dict(x=[0.55, 1.0], y=[0.55, 1.0]),
        marker=dict(colors=['#3498DB', '#9B59B6', '#F1C40F'])
    ))
    
    # Add time-based heatmap
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Create sample heatmap data
    np.random.seed(0)
    z = np.random.normal(10, 2, size=(7, 24))
    
    # Add patterns (higher in work hours on weekdays)
    for day in range(5):  # Weekdays
        for hour in range(8, 18):  # Business hours
            z[day, hour] += 5
    
    # Add a few anomalies
    z[1, 14] += 10  # Tuesday afternoon
    z[3, 10] += 12  # Thursday morning
    z[5, 22] += 8   # Saturday night
    
    fig.add_trace(go.Heatmap(
        z=z,
        x=hours,
        y=days,
        colorscale='Viridis'
    ))
    
    # Update the layout to position the heatmap in the lower part of the figure
    fig.update_layout(
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(domain=[0, 0.45])
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=20, b=20),
        height=550,
        annotations=[
            dict(text="Anomaly Distribution", x=0.225, y=0.775, showarrow=False, font_size=14),
            dict(text="Time Distribution", x=0.775, y=0.775, showarrow=False, font_size=14),
            dict(text="Energy Consumption Heatmap", x=0.5, y=0.225, showarrow=False, font_size=14)
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Call to action buttons
st.markdown("### Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Upload Data", use_container_width=True):
        st.switch_page("pages/03_upload_data.py")

with col2:
    if st.button("Run Detection", use_container_width=True):
        st.switch_page("pages/04_run_detection.py")

with col3:
    if st.button("View Dashboard", use_container_width=True):
        st.switch_page("pages/02_dashboard.py")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
