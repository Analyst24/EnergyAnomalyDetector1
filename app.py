import streamlit as st
import os
import json
import sys
from streamlit_option_menu import option_menu
import pandas as pd
import time
import warnings

# Import offline utilities first to enable offline mode from startup
from utils.offline import OFFLINE_MODE, log_offline_activity, is_connected
from utils.auth import login_user, register_user, logout_user
from utils.database import initialize_database

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up page configuration - making sidebar collapsed for login page
st.set_page_config(
    page_title="Energy Anomaly Detection (Offline)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",  # Changed to collapsed for login page
)

# Log application start
log_offline_activity("app_startup", {
    "python_version": sys.version,
    "offline_mode": OFFLINE_MODE,
    "internet_connection": is_connected()
})

# Initialize database
initialize_database()

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "detection_results" not in st.session_state:
    st.session_state.detection_results = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "isolation_forest"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

# Authentication functions using database
def login(username, password):
    return login_user(username, password)

def signup(username, email, password):
    return register_user(username, email, password)

def logout():
    logout_user()
    st.session_state.current_data = None
    st.session_state.detection_results = None

# Display offline mode status
if OFFLINE_MODE:
    st.markdown(
        """
        <div style='
            padding: 0.5rem; 
            background-color: #ff9900; 
            color: black; 
            position: fixed; 
            top: 0; 
            right: 0; 
            z-index: 9999; 
            font-size: 0.8rem; 
            border-radius: 0 0 0 5px;'>
            OFFLINE MODE
        </div>
        """,
        unsafe_allow_html=True
    )

# Login UI
if not st.session_state.authenticated:
    # Hide sidebar completely for login page
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        .login-container {
            background-color: #252525;
            padding: 30px;
            border-radius: 10px;
            margin-top: 50px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        with st.container():
            st.markdown("<h1 style='text-align: center;'>Energy Anomaly Detection</h1>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            
            with tab1:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_button = st.button("Login", use_container_width=True)
                
                if login_button:
                    if login(username, password):
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with tab2:
                new_username = st.text_input("Username", key="new_username")
                email = st.text_input("Email")
                new_password = st.text_input("Password", type="password", key="new_password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                signup_button = st.button("Sign Up", use_container_width=True)
                
                if signup_button:
                    if not new_username or not email or not new_password:
                        st.error("All fields are required")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif signup(new_username, email, new_password):
                        st.success("Account created successfully! Please login.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to create account. Username may already exist.")
    
    # Footer
    st.markdown(
        """
        <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1E1E1E;'>
        © 2025 Opulent Chikwiramakomo. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Redirect to Home page after login
    st.switch_page("pages/01_home.py")
