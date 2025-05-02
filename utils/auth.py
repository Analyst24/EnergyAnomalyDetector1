import hashlib
import json
import os
import time
import secrets
import streamlit as st

# Path to users file (for demo purposes)
USERS_FILE = "data/users.json"

def initialize_users_file():
    """Initialize users file if it doesn't exist"""
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump({
                "demo": {
                    "password_hash": hash_password("energy123"),
                    "email": "demo@example.com",
                    "created_at": time.time()
                }
            }, f)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    initialize_users_file()
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        # If file is corrupted or can't be loaded, initialize it again
        initialize_users_file()
        with open(USERS_FILE, 'r') as f:
            return json.load(f)

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def verify_user(username, password):
    """Verify user credentials"""
    users = load_users()
    
    if username in users:
        if users[username]["password_hash"] == hash_password(password):
            return True
    
    return False

def register_user(username, email, password):
    """Register a new user"""
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False
    
    # Create new user
    users[username] = {
        "password_hash": hash_password(password),
        "email": email,
        "created_at": time.time()
    }
    
    save_users(users)
    return True

def login_user(username, password):
    """Log in a user and set session state"""
    if verify_user(username, password):
        st.session_state.authenticated = True
        st.session_state.username = username
        return True
    return False

def logout_user():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.username = ""

def require_auth():
    """Check if user is authenticated, redirect to login if not"""
    if not st.session_state.get("authenticated", False):
        st.switch_page("app.py")
