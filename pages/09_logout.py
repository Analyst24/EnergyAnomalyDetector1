import streamlit as st
import time
from streamlit_option_menu import option_menu
from utils.auth import logout_user

# Set page config
st.set_page_config(
    page_title="Logout - Energy Anomaly Detection",
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
        default_index=8,
    )
    
    # Handle navigation (if user hasn't logged out)
    if selected != "Logout" and st.session_state.get("authenticated", False):
        if selected == "Home":
            st.switch_page("pages/01_get_started.py")
        else:
            page_name = selected.lower().replace(' ', '_')
            idx = ["dashboard", "upload_data", "run_detection", "results", 
                  "model_insights", "recommendations", "settings"].index(page_name)
            st.switch_page(f"pages/0{idx + 2}_{page_name}.py")

# Main container
st.markdown("<h1 style='text-align: center;'>Log Out</h1>", unsafe_allow_html=True)

# Check if user is authenticated
if st.session_state.get("authenticated", False):
    # Display logout confirmation
    st.markdown(f"### Are you sure you want to log out, {st.session_state.username}?")
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create logout button
        if st.button("Log Out", use_container_width=True):
            # Show spinner while logging out
            with st.spinner("Logging out..."):
                # Clear session state
                logout_user()
                
                # Short delay for user experience
                time.sleep(1)
            
            # Show success message
            st.success("You have been logged out successfully.")
            
            # Redirect to login page
            time.sleep(1)
            st.switch_page("app.py")
        
        # Cancel button
        if st.button("Cancel", use_container_width=True):
            # Redirect to home page
            st.switch_page("pages/01_get_started.py")
else:
    # User is already logged out
    st.info("You are not currently logged in.")
    
    # Create login button
    if st.button("Go to Login", use_container_width=True):
        st.switch_page("app.py")

# Footer
st.markdown(
    """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
    © 2025 Opulent Chikwiramakomo. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
