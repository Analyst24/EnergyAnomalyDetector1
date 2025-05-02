import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import time
import plotly.express as px
from streamlit_option_menu import option_menu
from utils.auth import require_auth
from utils.data_processor import load_data, preprocess_data, check_data_quality, get_data_summary

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Upload Data - Energy Anomaly Detection",
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
        default_index=2,
    )
    
    # Handle navigation
    if selected != "Upload Data":
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "run_detection", "results", "model_insights", 
                  "recommendations", "settings", "logout"].index(page_name)
            st.switch_page(f"pages/0{idx + 2}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Upload Energy Data</h1>", unsafe_allow_html=True)

# Upload widget
st.markdown("### Select a CSV file to upload")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Process the uploaded file
if uploaded_file is not None:
    try:
        with st.spinner("Processing the uploaded data..."):
            # Read the file as a dataframe
            df = pd.read_csv(uploaded_file)
            
            # Display basic information
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            
            # Check if dataframe is empty
            if df.empty:
                st.error("The uploaded file is empty. Please upload a valid CSV file.")
            else:
                # Display dataframe info
                st.markdown("### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check required columns
                required_columns = ['timestamp']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"Missing recommended columns: {', '.join(missing_columns)}. Some features may be limited.")
                
                # Process data
                with st.spinner("Preprocessing and analyzing data..."):
                    # Preprocess the data
                    df_processed = preprocess_data(df)
                    
                    # Check data quality
                    quality_report = check_data_quality(df_processed)
                    
                    # Get data summary
                    data_summary = get_data_summary(df_processed)
                    
                    # Store processed data in session state
                    st.session_state.current_data = df_processed
                    
                    # Clear any previous results
                    if 'detection_results' in st.session_state:
                        st.session_state.detection_results = None
                
                # Display data quality summary
                st.markdown("### Data Quality Summary")
                
                # Create metrics row
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Total Rows", quality_report['row_count'])
                
                with metric_cols[1]:
                    st.metric("Total Columns", quality_report['column_count'])
                
                with metric_cols[2]:
                    st.metric("Duplicate Rows", quality_report['duplicates'])
                
                with metric_cols[3]:
                    missing_count = sum(quality_report['missing_values'].values())
                    st.metric("Missing Values", missing_count)
                
                # Create tabs for detailed information
                tab1, tab2, tab3 = st.tabs(["Data Structure", "Missing Values", "Data Distribution"])
                
                with tab1:
                    # Show column types
                    st.markdown("#### Column Types")
                    dtypes_df = pd.DataFrame({
                        'Column': df_processed.columns,
                        'Type': df_processed.dtypes.astype(str),
                        'Non-Null Count': df_processed.count().values,
                        'Null Count': df_processed.isnull().sum().values,
                        'Null Percentage': (df_processed.isnull().sum().values / len(df_processed) * 100).round(2)
                    })
                    st.dataframe(dtypes_df, use_container_width=True)
                
                with tab2:
                    # Show missing values
                    st.markdown("#### Missing Values by Column")
                    
                    # Create missing values dataframe
                    missing_df = pd.DataFrame({
                        'Column': list(quality_report['missing_values'].keys()),
                        'Missing Count': list(quality_report['missing_values'].values()),
                        'Missing Percentage': list(quality_report['missing_percentage'].values())
                    })
                    
                    # Sort by missing count
                    missing_df = missing_df.sort_values('Missing Count', ascending=False)
                    
                    # Display as dataframe
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Create bar chart for missing values
                    if not missing_df.empty and missing_df['Missing Count'].sum() > 0:
                        fig = px.bar(
                            missing_df[missing_df['Missing Count'] > 0],
                            x='Column',
                            y='Missing Percentage',
                            color='Missing Percentage',
                            color_continuous_scale='Reds',
                            template='plotly_dark',
                            title='Missing Values by Column (%)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No missing values in the dataset.")
                
                with tab3:
                    # Show data distribution
                    st.markdown("#### Numeric Columns Distribution")
                    
                    # If we have numeric columns, create distribution plots
                    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 0:
                        # Let user select column to visualize
                        selected_col = st.selectbox("Select column:", numeric_cols)
                        
                        # Create histogram
                        fig = px.histogram(
                            df_processed,
                            x=selected_col,
                            marginal='box',
                            template='plotly_dark',
                            title=f'Distribution of {selected_col}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show statistics
                        st.markdown("#### Summary Statistics")
                        stats = df_processed[selected_col].describe()
                        st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
                    else:
                        st.info("No numeric columns to analyze.")
                
                # Show inconsistencies if any
                if quality_report['inconsistencies']:
                    st.markdown("### Data Inconsistencies")
                    for inconsistency in quality_report['inconsistencies']:
                        st.warning(
                            f"{inconsistency['type']}: {inconsistency['count']} instances"
                        )
                
                # Buttons for next steps
                col1, col2 = st.columns(2)
                
                with col1:
                    # Save a sample dataset
                    save_button = st.button("Save Data", use_container_width=True)
                    if save_button:
                        # Save to session state
                        st.session_state.current_data = df_processed
                        st.success("Data saved successfully!")
                
                with col2:
                    proceed_button = st.button("Proceed to Anomaly Detection", use_container_width=True)
                    if proceed_button:
                        # Save to session state and redirect
                        st.session_state.current_data = df_processed
                        st.switch_page("pages/04_run_detection.py")
    
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    # Show sample data template when no file is uploaded
    st.info("No file uploaded. Upload a CSV file or use the template below.")
    
    # Sample data template
    st.markdown("### Sample Data Template")
    st.markdown("""
    Your CSV file should ideally contain the following columns:
    - `timestamp`: Date and time of the measurement (required)
    - `consumption`: Energy consumption value
    - `meter_id`: Identifier for the energy meter
    - `location`: Location of the meter
    - `temperature`: Ambient temperature (optional)
    - `humidity`: Ambient humidity (optional)
    - `season` or `time_of_day`: Contextual time information (optional)
    
    The system can handle missing values and will adapt to your data structure.
    """)
    
    # Sample data download
    st.markdown("### Download Sample Dataset")
    
    # Create sample data
    def generate_sample_data():
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=1000, freq='H')
        
        # Create consumption with daily and weekly patterns
        hour_effect = np.sin(np.pi * dates.hour / 24) * 10
        day_effect = np.sin(np.pi * dates.dayofweek / 7) * 5
        
        consumption = 100 + hour_effect + day_effect
        
        # Add trend and noise
        consumption += np.linspace(0, 10, len(dates))
        consumption += np.random.normal(0, 5, len(dates))
        
        # Create anomalies (for documentation, not detected yet)
        anomaly_indices = np.random.choice(range(len(dates)), size=50, replace=False)
        for idx in anomaly_indices:
            consumption[idx] *= np.random.choice([0.5, 1.5])
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'consumption': consumption,
            'meter_id': np.random.choice(['M001', 'M002', 'M003', 'M004', 'M005'], len(dates)),
            'location': np.random.choice(['Office', 'Factory', 'Warehouse', 'Data Center'], len(dates)),
            'temperature': 20 + 10 * np.sin(np.pi * np.arange(len(dates)) / (24 * 30)) + np.random.normal(0, 2, len(dates)),
            'humidity': 50 + 20 * np.sin(np.pi * np.arange(len(dates)) / (24 * 15)) + np.random.normal(0, 5, len(dates))
        })
        
        return df
    
    sample_df = generate_sample_data()
    
    # Show preview
    st.dataframe(sample_df.head(), use_container_width=True)
    
    # Create download button
    csv = sample_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_energy_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Instructions for using the sample data
    st.markdown("""
    **To use the sample data:**
    1. Click the "Download Sample CSV" button above
    2. Upload the downloaded file using the file uploader at the top of this page
    3. Proceed with the anomaly detection process
    """)

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
