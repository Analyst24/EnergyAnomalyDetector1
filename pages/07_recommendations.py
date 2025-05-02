import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from utils.auth import require_auth
from utils.helpers import generate_recommendations

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Recommendations - Energy Anomaly Detection",
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
        default_index=6,
    )
    
    # Handle navigation
    if selected != "Recommendations":
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "upload_data", "run_detection", "results", 
                  "model_insights", "settings", "logout"].index(page_name)
            if idx < 5:
                st.switch_page(f"pages/0{idx + 2}_{page_name}.py")
            else:
                st.switch_page(f"pages/0{idx + 3}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Energy Efficiency Recommendations</h1>", unsafe_allow_html=True)

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
    st.markdown("### Anomaly Summary")
    
    # Create metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Anomalies Detected", metadata.get('anomaly_count', 'N/A'))
    
    with col2:
        percentage = metadata.get('anomaly_percentage', 'N/A')
        if percentage != 'N/A':
            percentage = f"{percentage:.2f}%"
        st.metric("Anomaly Percentage", percentage)
    
    with col3:
        if 'timestamp' in result_df.columns and 'anomaly' in result_df.columns:
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            result_df['hour'] = result_df['timestamp'].dt.hour
            peak_hour = result_df.groupby('hour')['anomaly'].sum().idxmax()
            st.metric("Peak Anomaly Hour", f"{peak_hour}:00")
        else:
            st.metric("Peak Anomaly Hour", "N/A")
    
    # Generate recommendations
    recommendations = generate_recommendations(result_df)
    
    # Display recommendations
    st.markdown("### Key Recommendations")
    
    # Create expandable sections for recommendation categories
    with st.expander("Energy Usage Optimization", expanded=True):
        st.markdown("""
        <div style="padding: 10px; border-left: 4px solid #2ECC71;">
        <h4>Energy Usage Patterns</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations[:3]):
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 10px; background-color: #252525; border-radius: 5px;">
            <span style="color: #2ECC71; font-weight: bold;">✓</span> {rec}
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("Anomaly Investigation", expanded=True):
        st.markdown("""
        <div style="padding: 10px; border-left: 4px solid #3498DB;">
        <h4>Addressing Detected Anomalies</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations[3:6]):
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 10px; background-color: #252525; border-radius: 5px;">
            <span style="color: #3498DB; font-weight: bold;">→</span> {rec}
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("System Improvements", expanded=True):
        st.markdown("""
        <div style="padding: 10px; border-left: 4px solid #F39C12;">
        <h4>Long-term Efficiency Measures</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations[6:]):
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 10px; background-color: #252525; border-radius: 5px;">
            <span style="color: #F39C12; font-weight: bold;">⚡</span> {rec}
            </div>
            """, unsafe_allow_html=True)
    
    # Time-based recommendations if available
    if 'timestamp' in result_df.columns and 'anomaly' in result_df.columns:
        st.markdown("### Time-based Analysis")
        
        # Process time data
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        result_df['hour'] = result_df['timestamp'].dt.hour
        result_df['day_of_week'] = result_df['timestamp'].dt.dayofweek
        
        # Group by hour
        hourly_data = result_df.groupby('hour').agg({
            'anomaly': ['sum', 'mean'],
            'consumption': 'mean' if 'consumption' in result_df.columns else 'count'
        }).reset_index()
        
        hourly_data.columns = ['hour', 'anomaly_count', 'anomaly_rate', 'avg_consumption']
        hourly_data['anomaly_rate'] = hourly_data['anomaly_rate'] * 100
        
        # Create heatmap for time recommendations
        # Define business hours and off hours
        business_hours = list(range(8, 18))  # 8 AM to 6 PM
        night_hours = list(range(0, 6)) + list(range(22, 24))  # 10 PM to 6 AM
        
        # Assign categories
        hourly_data['time_category'] = 'Other'
        hourly_data.loc[hourly_data['hour'].isin(business_hours), 'time_category'] = 'Business Hours'
        hourly_data.loc[hourly_data['hour'].isin(night_hours), 'time_category'] = 'Night Hours'
        
        # Create stacked bar chart
        fig1 = px.bar(
            hourly_data,
            x='hour',
            y='anomaly_count',
            color='time_category',
            color_discrete_map={
                'Business Hours': '#3498DB',
                'Night Hours': '#9B59B6',
                'Other': '#2ECC71'
            },
            labels={
                'hour': 'Hour of Day',
                'anomaly_count': 'Number of Anomalies',
                'time_category': 'Time Category'
            },
            title='Anomalies by Hour of Day',
            template='plotly_dark'
        )
        
        # Add line for anomaly rate
        fig1.add_trace(
            go.Scatter(
                x=hourly_data['hour'],
                y=hourly_data['anomaly_rate'],
                mode='lines+markers',
                name='Anomaly Rate (%)',
                yaxis='y2',
                line=dict(color='#F39C12', width=2)
            )
        )
        
        # Update layout
        fig1.update_layout(
            xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
            yaxis=dict(title='Number of Anomalies'),
            yaxis2=dict(
                title='Anomaly Rate (%)',
                overlaying='y',
                side='right',
                range=[0, max(hourly_data['anomaly_rate']) * 1.2]
            ),
            legend=dict(orientation='h', y=1.1),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Display chart
        st.plotly_chart(fig1, use_container_width=True)
        
        # Time-based recommendations
        st.markdown("#### Time-specific Recommendations")
        
        # Get peak hours for anomalies
        peak_hour = hourly_data.loc[hourly_data['anomaly_count'].idxmax(), 'hour']
        
        # Business hours recommendations
        business_anomalies = hourly_data[hourly_data['time_category'] == 'Business Hours']
        if not business_anomalies.empty:
            peak_business_hour = business_anomalies.loc[business_anomalies['anomaly_count'].idxmax(), 'hour']
            
            st.markdown(f"""
            <div style="padding: 15px; margin-bottom: 15px; background-color: #1A5276; border-radius: 5px; border-left: 5px solid #3498DB;">
            <h4>Business Hours (8 AM - 6 PM)</h4>
            <p>Peak anomaly hour during business: <b>{peak_business_hour}:00</b></p>
            <ul>
                <li>Schedule energy audits during peak anomaly hours to identify causes</li>
                <li>Implement load balancing to reduce peak demand charges</li>
                <li>Review HVAC setpoints and schedules to optimize for occupancy patterns</li>
                <li>Consider installing motion sensors for lighting in less frequently used areas</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Night hours recommendations
        night_anomalies = hourly_data[hourly_data['time_category'] == 'Night Hours']
        if not night_anomalies.empty:
            peak_night_hour = night_anomalies.loc[night_anomalies['anomaly_count'].idxmax(), 'hour']
            
            st.markdown(f"""
            <div style="padding: 15px; margin-bottom: 15px; background-color: #5B2C6F; border-radius: 5px; border-left: 5px solid #9B59B6;">
            <h4>Night Hours (10 PM - 6 AM)</h4>
            <p>Peak anomaly hour during night: <b>{peak_night_hour}:00</b></p>
            <ul>
                <li>Investigate unexpected energy usage during off-hours</li>
                <li>Implement automatic shutdown procedures for non-critical systems</li>
                <li>Check for equipment left running overnight unnecessarily</li>
                <li>Consider programmable timers for equipment that must run intermittently</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Location-based recommendations if available
    if 'location' in result_df.columns and 'anomaly' in result_df.columns:
        st.markdown("### Location-based Analysis")
        
        # Group by location
        location_data = result_df.groupby('location').agg({
            'anomaly': ['sum', 'mean'],
            'consumption': 'mean' if 'consumption' in result_df.columns else 'count'
        }).reset_index()
        
        location_data.columns = ['location', 'anomaly_count', 'anomaly_rate', 'avg_consumption']
        location_data['anomaly_rate'] = location_data['anomaly_rate'] * 100
        
        # Sort by anomaly count
        location_data = location_data.sort_values('anomaly_count', ascending=False)
        
        # Create bar chart
        fig2 = px.bar(
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
        
        fig2.update_layout(
            xaxis_title='Location',
            yaxis_title='Number of Anomalies',
            coloraxis_colorbar_title='Anomaly Rate (%)'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Location-based recommendations
        st.markdown("#### Location-specific Recommendations")
        
        # Get locations with highest anomaly rates
        top_locations = location_data.head(3)
        
        for _, loc in top_locations.iterrows():
            location_name = loc['location']
            anomaly_count = int(loc['anomaly_count'])
            anomaly_rate = loc['anomaly_rate']
            
            st.markdown(f"""
            <div style="padding: 15px; margin-bottom: 15px; background-color: #145A32; border-radius: 5px; border-left: 5px solid #2ECC71;">
            <h4>Location: {location_name}</h4>
            <p>Anomalies: <b>{anomaly_count}</b> (Rate: <b>{anomaly_rate:.1f}%</b>)</p>
            <ul>
                <li>Conduct a detailed energy audit for this location</li>
                <li>Check for equipment malfunctions or inefficient operation</li>
                <li>Compare usage patterns with similar locations</li>
                <li>Consider priority replacement of older equipment</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Efficiency calculator
    st.markdown("### Efficiency Savings Calculator")
    
    # Create calculator form
    with st.form("savings_calculator"):
        st.markdown("#### Estimate potential savings from addressing anomalies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_consumption = 100
            if 'consumption' in result_df.columns:
                avg_consumption = result_df['consumption'].mean()
            
            daily_consumption = st.number_input(
                "Average Daily Energy Consumption (kWh)",
                min_value=1.0,
                value=float(avg_consumption) * 24 if avg_consumption > 0 else 1000.0,
                step=100.0
            )
            
            energy_cost = st.number_input(
                "Energy Cost ($/kWh)",
                min_value=0.01,
                value=0.15,
                step=0.01
            )
            
            anomaly_reduction = st.slider(
                "Expected Anomaly Reduction (%)",
                min_value=10,
                max_value=90,
                value=50,
                step=10
            )
        
        with col2:
            energy_savings = st.slider(
                "Estimated Energy Savings per Anomaly (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )
            
            implementation_cost = st.number_input(
                "Implementation Cost ($)",
                min_value=0,
                value=1000,
                step=100
            )
            
            time_period = st.selectbox(
                "Calculation Period",
                options=["Monthly", "Quarterly", "Yearly"],
                index=2
            )
        
        # Calculate button
        calculate_button = st.form_submit_button("Calculate Savings", use_container_width=True)
    
    # Display results if button clicked
    if calculate_button or 'savings_calculated' in st.session_state:
        # Store calculation state
        st.session_state.savings_calculated = True
        
        # Time multiplier
        time_multipliers = {"Monthly": 30, "Quarterly": 90, "Yearly": 365}
        days = time_multipliers[time_period]
        
        # Calculate anomaly impact
        anomaly_count = metadata.get('anomaly_count', 0)
        if anomaly_count == 0:
            anomaly_count = int(daily_consumption * 0.05)  # Assume 5% anomaly rate if no data
        
        reduced_anomalies = anomaly_count * (anomaly_reduction / 100)
        
        # Calculate energy savings
        energy_saved_daily = daily_consumption * (energy_savings / 100) * (reduced_anomalies / daily_consumption)
        energy_saved_period = energy_saved_daily * days
        
        # Calculate cost savings
        cost_savings = energy_saved_period * energy_cost
        
        # Calculate ROI
        if implementation_cost > 0:
            roi = (cost_savings - implementation_cost) / implementation_cost * 100
            payback_days = implementation_cost / (energy_saved_daily * energy_cost)
            
            if time_period == "Monthly":
                payback_period = f"{payback_days:.1f} days"
                if payback_days > 30:
                    payback_period = f"{payback_days/30:.1f} months"
            elif time_period == "Quarterly":
                payback_period = f"{payback_days:.1f} days"
                if payback_days > 90:
                    payback_period = f"{payback_days/90:.1f} quarters"
            else:
                payback_period = f"{payback_days:.1f} days"
                if payback_days > 365:
                    payback_period = f"{payback_days/365:.1f} years"
        else:
            roi = float('inf')
            payback_period = "Immediate"
        
        # Display results
        st.markdown("#### Savings Estimate Results")
        
        # Create cards for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="padding: 20px; text-align: center; background-color: #252525; border-radius: 10px; height: 150px;">
            <h3 style="margin-bottom: 10px; color: #3498DB;">Energy Savings</h3>
            <p style="font-size: 28px; font-weight: bold;">{energy_saved_period:.2f} kWh</p>
            <p>per {time_period.lower()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; text-align: center; background-color: #252525; border-radius: 10px; height: 150px;">
            <h3 style="margin-bottom: 10px; color: #2ECC71;">Cost Savings</h3>
            <p style="font-size: 28px; font-weight: bold;">${cost_savings:.2f}</p>
            <p>per {time_period.lower()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            roi_color = "#2ECC71" if roi > 0 else "#E74C3C"
            st.markdown(f"""
            <div style="padding: 20px; text-align: center; background-color: #252525; border-radius: 10px; height: 150px;">
            <h3 style="margin-bottom: 10px; color: {roi_color};">ROI</h3>
            <p style="font-size: 28px; font-weight: bold;">{roi:.1f}%</p>
            <p>Payback: {payback_period}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional information
        st.markdown("""
        **Note:** These calculations are estimates based on the provided inputs and detected anomalies. 
        Actual savings may vary depending on implementation details and other factors.
        """)
    
    # Call to action
    st.markdown("### Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("View Detailed Results", use_container_width=True):
            st.switch_page("pages/05_results.py")
    
    with col2:
        if st.button("Run New Detection", use_container_width=True):
            st.switch_page("pages/04_run_detection.py")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
