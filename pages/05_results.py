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
    create_feature_correlation,
    create_anomaly_scores_histogram
)
from utils.helpers import export_csv, export_figure_as_image, generate_pdf_report

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Results - Energy Anomaly Detection",
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
        default_index=4,
    )
    
    # Handle navigation
    if selected != "Results":
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "upload_data", "run_detection", "model_insights", 
                  "recommendations", "settings", "logout"].index(page_name)
            if idx < 3:
                st.switch_page(f"pages/0{idx + 2}_{page_name}.py")
            else:
                st.switch_page(f"pages/0{idx + 3}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Detection Results</h1>", unsafe_allow_html=True)

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
    
    # Display summary metrics
    st.markdown("### Detection Summary")
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Anomalies Detected", metadata.get('anomaly_count', 'N/A'))
    
    with col2:
        st.metric("Normal Points", metadata.get('normal_count', 'N/A'))
    
    with col3:
        percentage = metadata.get('anomaly_percentage', 'N/A')
        if percentage != 'N/A':
            percentage = f"{percentage:.2f}%"
        st.metric("Anomaly Percentage", percentage)
    
    with col4:
        model_name = metadata.get('model_type', 'unknown')
        model_display = {
            'isolation_forest': 'Isolation Forest',
            'kmeans': 'K-Means',
            'autoencoder': 'Autoencoder'
        }.get(model_name, model_name)
        st.metric("Model Used", model_display)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Analysis", "Feature Analysis", "Data Table"])
    
    with tab1:
        # Energy consumption overview with anomalies
        st.markdown("### Energy Consumption with Anomalies")
        
        # Check if we have time series data
        has_time_series = 'timestamp' in result_df.columns and 'consumption' in result_df.columns
        
        if has_time_series:
            # Create and display the overview chart
            fig1 = create_energy_overview(result_df)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Export chart
            st.markdown("Export chart:")
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.markdown(export_figure_as_image(fig1, "energy_overview.png", "png"), unsafe_allow_html=True)
            
            with export_col2:
                st.markdown(export_figure_as_image(fig1, "energy_overview.svg", "svg"), unsafe_allow_html=True)
        else:
            st.info("Time series data (timestamp and consumption) not available for this visualization.")
        
        # Anomaly distribution
        st.markdown("### Anomaly Distribution")
        fig2 = create_anomaly_distribution(result_df)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Anomaly score histogram
        st.markdown("### Anomaly Score Distribution")
        fig3 = create_anomaly_scores_histogram(result_df)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        # Time-based analysis
        st.markdown("### Time of Day Analysis")
        
        if 'timestamp' in result_df.columns:
            # Make sure timestamp is datetime
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            
            # Time of day chart
            fig4 = create_hourly_analysis(result_df)
            st.plotly_chart(fig4, use_container_width=True)
            
            # Day of week analysis
            st.markdown("### Day of Week Analysis")
            
            # Create day of week if it doesn't exist
            if 'day_of_week' not in result_df.columns:
                result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek
            
            # Group by day of week
            day_data = result_df.groupby('day_of_week')['anomaly'].agg(['sum', 'count']).reset_index()
            day_data['normal'] = day_data['count'] - day_data['sum']
            day_data['anomaly_rate'] = day_data['sum'] / day_data['count'] * 100
            day_data['day_name'] = day_data['day_of_week'].map({
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            })
            
            # Create figure
            fig5 = go.Figure()
            
            # Add bar chart for counts
            fig5.add_trace(
                go.Bar(x=day_data['day_name'], y=day_data['normal'], name='Normal',
                      marker_color='#2ECC71')
            )
            fig5.add_trace(
                go.Bar(x=day_data['day_name'], y=day_data['sum'], name='Anomalies',
                      marker_color='#E74C3C')
            )
            
            # Add line chart for anomaly rate
            fig5.add_trace(
                go.Scatter(x=day_data['day_name'], y=day_data['anomaly_rate'], name='Anomaly Rate (%)',
                          mode='lines+markers', yaxis='y2',
                          line=dict(color='#F39C12', width=2))
            )
            
            # Update layout
            fig5.update_layout(
                title='Anomalies by Day of Week',
                barmode='stack',
                template='plotly_dark',
                yaxis=dict(title='Number of Data Points'),
                yaxis2=dict(title='Anomaly Rate (%)', overlaying='y', side='right',
                           range=[0, max(day_data['anomaly_rate']) * 1.2]),
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig5, use_container_width=True)
            
            # Monthly analysis if we have enough data
            if 'month' not in result_df.columns:
                result_df['month'] = result_df['timestamp'].dt.month
                
            # Check if we have multiple months
            if result_df['month'].nunique() > 1:
                st.markdown("### Monthly Analysis")
                
                # Group by month
                month_data = result_df.groupby('month')['anomaly'].agg(['sum', 'count']).reset_index()
                month_data['normal'] = month_data['count'] - month_data['sum']
                month_data['anomaly_rate'] = month_data['sum'] / month_data['count'] * 100
                month_data['month_name'] = month_data['month'].map({
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                })
                
                # Create figure
                fig6 = px.line(
                    month_data, x='month_name', y='anomaly_rate',
                    markers=True,
                    labels={'month_name': 'Month', 'anomaly_rate': 'Anomaly Rate (%)'},
                    title='Monthly Anomaly Rate',
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Time-based analysis requires timestamp data, which is not available in the current dataset.")
    
    with tab3:
        # Feature analysis
        st.markdown("### Feature Correlation")
        
        fig7 = create_feature_correlation(result_df)
        st.plotly_chart(fig7, use_container_width=True)
        
        # Feature importance if available
        if 'feature_importance' in metadata:
            st.markdown("### Feature Importance")
            
            # Get feature importance
            importance = metadata['feature_importance']
            features = list(importance.keys())
            values = list(importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            sorted_features = [features[i] for i in sorted_idx]
            sorted_values = [values[i] for i in sorted_idx]
            
            # Create bar chart
            fig8 = px.bar(
                x=sorted_values,
                y=sorted_features,
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title='Feature Importance',
                template='plotly_dark'
            )
            
            fig8.update_layout(
                xaxis_title='Importance',
                yaxis_title='Feature',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig8, use_container_width=True)
        
        # Location-based analysis if available
        if 'location' in result_df.columns:
            st.markdown("### Location-based Analysis")
            
            # Group by location
            location_data = result_df.groupby('location').agg({
                'anomaly': ['sum', 'mean'],
                'consumption': 'mean' if 'consumption' in result_df.columns else 'count'
            }).reset_index()
            
            location_data.columns = ['location', 'anomaly_count', 'anomaly_rate', 'avg_consumption']
            location_data['anomaly_rate'] = location_data['anomaly_rate'] * 100
            
            # Create bar chart
            fig9 = px.bar(
                location_data,
                x='location',
                y='anomaly_count',
                color='anomaly_rate',
                hover_data=['avg_consumption'],
                labels={
                    'location': 'Location',
                    'anomaly_count': 'Number of Anomalies',
                    'anomaly_rate': 'Anomaly Rate (%)',
                    'avg_consumption': 'Avg. Consumption'
                },
                title='Anomalies by Location',
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            
            fig9.update_layout(
                xaxis_title='Location',
                yaxis_title='Number of Anomalies',
                coloraxis_colorbar_title='Anomaly Rate (%)'
            )
            
            st.plotly_chart(fig9, use_container_width=True)
    
    with tab4:
        # Data table with anomalies
        st.markdown("### Data Table with Anomalies")
        
        # Add filter for showing only anomalies
        show_only_anomalies = st.checkbox("Show only anomalies", value=False)
        
        # Filter data if needed
        display_df = result_df
        if show_only_anomalies and 'anomaly' in result_df.columns:
            display_df = result_df[result_df['anomaly'] == 1]
        
        # Show the dataframe
        st.dataframe(display_df, use_container_width=True)
        
        # Allow export of results
        st.markdown("### Export Results")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown(export_csv(display_df, "anomaly_results.csv"), unsafe_allow_html=True)
        
        with export_col2:
            # Generate PDF report
            pdf_data = generate_pdf_report(result_df, metadata)
            
            # Create download button for PDF
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name="anomaly_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    # Action buttons
    st.markdown("### Next Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run New Detection", use_container_width=True):
            st.switch_page("pages/04_run_detection.py")
    
    with col2:
        if st.button("View Model Insights", use_container_width=True):
            st.switch_page("pages/06_model_insights.py")
    
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
