import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from streamlit_option_menu import option_menu
from utils.auth import require_auth

# Require authentication
require_auth()

# Set page config
st.set_page_config(
    page_title="Settings - Energy Anomaly Detection",
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
        default_index=7,
    )
    
    # Handle navigation
    if selected != "Settings":
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "upload_data", "run_detection", "results", 
                  "model_insights", "recommendations", "logout"].index(page_name)
            if idx < 6:
                st.switch_page(f"pages/0{idx + 2}_{page_name}.py")
            else:
                st.switch_page(f"pages/0{idx + 3}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Settings</h1>", unsafe_allow_html=True)

# Initialize settings if they don't exist
if 'settings' not in st.session_state:
    # Default settings
    st.session_state.settings = {
        'default_model': 'isolation_forest',
        'default_threshold': 0.05,
        'notifications_enabled': True,
        'export_format': 'csv',
        'max_anomalies_display': 100,
        'chart_theme': 'dark'
    }

# Create tabs for different settings
tab1, tab2, tab3, tab4 = st.tabs(["Detection Settings", "Display Settings", "Data Management", "Account Settings"])

with tab1:
    st.markdown("### Anomaly Detection Settings")
    
    # Create form for detection settings
    with st.form("detection_settings"):
        # Default model
        default_model = st.selectbox(
            "Default Detection Model",
            options=["Isolation Forest", "K-Means Clustering", "Autoencoder Neural Network"],
            index=["Isolation Forest", "K-Means Clustering", "Autoencoder Neural Network"]
                .index("Isolation Forest" if st.session_state.settings['default_model'] == 'isolation_forest' else
                      "K-Means Clustering" if st.session_state.settings['default_model'] == 'kmeans' else
                      "Autoencoder Neural Network")
        )
        
        # Default threshold
        default_threshold = st.slider(
            "Default Anomaly Threshold",
            min_value=0.01,
            max_value=0.2,
            value=st.session_state.settings['default_threshold'],
            step=0.01,
            help="Lower values are more sensitive (detect more anomalies)"
        )
        
        # Advanced settings
        st.markdown("#### Advanced Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_scaling = st.checkbox("Automatically scale features", value=True)
            handle_missing = st.checkbox("Automatically handle missing values", value=True)
        
        with col2:
            save_models = st.checkbox("Save trained models for future use", value=True)
            cache_results = st.checkbox("Cache detection results", value=True)
        
        # Submit button
        detection_settings_submit = st.form_submit_button("Save Detection Settings", use_container_width=True)
    
    # Process form submission
    if detection_settings_submit:
        # Map model selection to internal name
        model_map = {
            "Isolation Forest": "isolation_forest",
            "K-Means Clustering": "kmeans",
            "Autoencoder Neural Network": "autoencoder"
        }
        
        # Update settings
        st.session_state.settings['default_model'] = model_map.get(default_model, "isolation_forest")
        st.session_state.settings['default_threshold'] = default_threshold
        st.session_state.settings['auto_scaling'] = auto_scaling
        st.session_state.settings['handle_missing'] = handle_missing
        st.session_state.settings['save_models'] = save_models
        st.session_state.settings['cache_results'] = cache_results
        
        # Show success message
        st.success("Detection settings saved successfully!")

with tab2:
    st.markdown("### Display Settings")
    
    # Create form for display settings
    with st.form("display_settings"):
        # Chart theme
        chart_theme = st.selectbox(
            "Chart Theme",
            options=["Dark", "Light"],
            index=0 if st.session_state.settings.get('chart_theme', 'dark') == 'dark' else 1,
            disabled=True  # Only dark theme is currently supported
        )
        
        if chart_theme == "Light":
            st.info("Light theme is planned for a future update. Currently, only dark theme is available.")
        
        # Visualization settings
        st.markdown("#### Visualization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_anomalies = st.number_input(
                "Maximum anomalies to display in tables",
                min_value=10,
                max_value=1000,
                value=st.session_state.settings.get('max_anomalies_display', 100),
                step=10
            )
            
            show_grid = st.checkbox("Show grid lines on charts", value=True)
        
        with col2:
            default_export = st.selectbox(
                "Default export format",
                options=["CSV", "PDF", "PNG"],
                index=["CSV", "PDF", "PNG"].index(st.session_state.settings.get('export_format', 'csv').upper())
            )
            
            show_tooltips = st.checkbox("Show tooltips on hover", value=True)
        
        # Submit button
        display_settings_submit = st.form_submit_button("Save Display Settings", use_container_width=True)
    
    # Process form submission
    if display_settings_submit:
        # Update settings
        st.session_state.settings['chart_theme'] = 'dark'  # Force dark theme for now
        st.session_state.settings['max_anomalies_display'] = max_anomalies
        st.session_state.settings['show_grid'] = show_grid
        st.session_state.settings['export_format'] = default_export.lower()
        st.session_state.settings['show_tooltips'] = show_tooltips
        
        # Show success message
        st.success("Display settings saved successfully!")
        
    # Preview section
    st.markdown("### Theme Preview")
    
    # Show theme preview
    st.markdown(f"""
    <div style="padding: 20px; background-color: #252525; border-radius: 10px; margin-top: 20px;">
    <h4>Dark Theme Preview</h4>
    <p>This is how your charts and visualizations will appear with the dark theme.</p>
    <div style="height: 150px; background: linear-gradient(90deg, #1E1E1E 0%, #252525 100%); border-radius: 5px; display: flex; align-items: center; justify-content: center; margin-top: 10px;">
    <span style="color: #FAFAFA;">Sample Visualization Area</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### Data Management")
    
    # File management
    st.markdown("#### Manage Uploaded Files")
    
    # Simulate list of uploaded files
    uploaded_files = [
        {"name": "energy_data_2025.csv", "size": "2.4 MB", "date": "2025-03-15"},
        {"name": "sample_energy_data.csv", "size": "1.2 MB", "date": "2025-03-10"},
    ]
    
    if len(uploaded_files) > 0:
        # Create dataframe for display
        files_df = pd.DataFrame(uploaded_files)
        st.dataframe(files_df, use_container_width=True)
        
        # Actions
        col1, col2 = st.columns(2)
        
        with col1:
            file_to_delete = st.selectbox(
                "Select file to delete",
                options=[file["name"] for file in uploaded_files]
            )
        
        with col2:
            if st.button("Delete Selected File", use_container_width=True):
                st.warning(f"This would delete {file_to_delete} in a production environment.")
    else:
        st.info("No files have been uploaded yet.")
    
    # Model management
    st.markdown("#### Manage Trained Models")
    
    # Simulate list of trained models
    trained_models = [
        {"name": "isolation_forest_2025-03-15", "type": "Isolation Forest", "accuracy": "94.2%", "date": "2025-03-15"},
        {"name": "kmeans_2025-03-10", "type": "K-Means", "accuracy": "91.5%", "date": "2025-03-10"},
    ]
    
    if len(trained_models) > 0:
        # Create dataframe for display
        models_df = pd.DataFrame(trained_models)
        st.dataframe(models_df, use_container_width=True)
        
        # Actions
        col1, col2 = st.columns(2)
        
        with col1:
            model_to_delete = st.selectbox(
                "Select model to delete",
                options=[model["name"] for model in trained_models]
            )
        
        with col2:
            if st.button("Delete Selected Model", use_container_width=True):
                st.warning(f"This would delete {model_to_delete} in a production environment.")
    else:
        st.info("No models have been trained yet.")
    
    # Data retention settings
    st.markdown("#### Data Retention Settings")
    
    with st.form("data_retention"):
        col1, col2 = st.columns(2)
        
        with col1:
            retain_data = st.slider(
                "Keep uploaded data for",
                min_value=1,
                max_value=12,
                value=3,
                step=1,
                format="%d months"
            )
        
        with col2:
            retain_models = st.slider(
                "Keep trained models for",
                min_value=1,
                max_value=12,
                value=6,
                step=1,
                format="%d months"
            )
        
        auto_cleanup = st.checkbox("Automatically delete expired data and models", value=True)
        
        # Submit button
        retention_submit = st.form_submit_button("Save Retention Settings", use_container_width=True)
    
    # Process form submission
    if retention_submit:
        st.success("Data retention settings saved successfully!")

with tab4:
    st.markdown("### Account Settings")
    
    # User information
    st.markdown("#### User Information")
    
    with st.form("user_info"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", value=st.session_state.username, disabled=True)
            email = st.text_input("Email", value="user@example.com")
        
        with col2:
            organization = st.text_input("Organization", value="Energy Company Inc.")
            role = st.text_input("Role", value="Energy Analyst")
        
        # Submit button
        user_info_submit = st.form_submit_button("Update User Information", use_container_width=True)
    
    # Process form submission
    if user_info_submit:
        st.success("User information updated successfully!")
    
    # Password change
    st.markdown("#### Change Password")
    
    with st.form("change_password"):
        col1, col2 = st.columns(2)
        
        with col1:
            current_password = st.text_input("Current Password", type="password")
        
        with col2:
            pass  # Empty column for alignment
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_password = st.text_input("New Password", type="password")
        
        with col2:
            confirm_password = st.text_input("Confirm New Password", type="password")
        
        # Submit button
        password_submit = st.form_submit_button("Change Password", use_container_width=True)
    
    # Process form submission
    if password_submit:
        if not current_password or not new_password or not confirm_password:
            st.error("All password fields are required.")
        elif new_password != confirm_password:
            st.error("New passwords do not match.")
        else:
            st.success("Password changed successfully!")
    
    # Notification settings
    st.markdown("#### Notification Settings")
    
    with st.form("notification_settings"):
        enable_notifications = st.checkbox(
            "Enable email notifications", 
            value=st.session_state.settings.get('notifications_enabled', True)
        )
        
        if enable_notifications:
            col1, col2 = st.columns(2)
            
            with col1:
                notify_anomalies = st.checkbox("Notify on significant anomalies", value=True)
                notify_model_complete = st.checkbox("Notify when model training completes", value=True)
            
            with col2:
                notify_reports = st.checkbox("Send weekly reports", value=True)
                notify_system = st.checkbox("System notifications", value=True)
        
        # Submit button
        notification_submit = st.form_submit_button("Save Notification Settings", use_container_width=True)
    
    # Process form submission
    if notification_submit:
        # Update settings
        st.session_state.settings['notifications_enabled'] = enable_notifications
        
        if enable_notifications:
            st.session_state.settings['notify_anomalies'] = notify_anomalies
            st.session_state.settings['notify_model_complete'] = notify_model_complete
            st.session_state.settings['notify_reports'] = notify_reports
            st.session_state.settings['notify_system'] = notify_system
        
        st.success("Notification settings saved successfully!")

# Reset button
st.markdown("### Reset All Settings")

if st.button("Reset to Default Settings", use_container_width=True):
    # Confirm reset
    st.warning("Are you sure you want to reset all settings to default values?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Yes, Reset Settings", key="confirm_reset", use_container_width=True):
            # Reset to defaults
            st.session_state.settings = {
                'default_model': 'isolation_forest',
                'default_threshold': 0.05,
                'notifications_enabled': True,
                'export_format': 'csv',
                'max_anomalies_display': 100,
                'chart_theme': 'dark'
            }
            
            st.success("Settings have been reset to default values.")
            st.rerun()
    
    with col2:
        if st.button("Cancel", key="cancel_reset", use_container_width=True):
            st.info("Reset cancelled.")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
