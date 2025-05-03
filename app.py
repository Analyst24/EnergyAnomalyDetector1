import streamlit as st
import os
import json
from streamlit_option_menu import option_menu
import pandas as pd
import time
from utils.auth import login_user, register_user, logout_user
from utils.database import initialize_database

# Set up page configuration
st.set_page_config(
    page_title="Energy Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Login UI
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <style>
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
    # Redirect to Get Started page after login
    st.switch_page("pages/01_get_started.py")
