import hashlib
import json
import os
import time
import secrets
import streamlit as st
from utils.database import add_user, get_user_by_username, Session, User

# Path to users file (for backward compatibility/fallback)
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
        
        # Also add demo user to database
        try:
            add_user("demo", "demo@example.com", hash_password("energy123"))
        except Exception as e:
            # User might already exist in database
            print(f"Note: {e}")

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file (legacy support)"""
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
    """Save users to JSON file (legacy support)"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def verify_user(username, password):
    """Verify user credentials using database"""
    # First try database
    try:
        user = get_user_by_username(username)
        if user and user.password_hash == hash_password(password):
            return True
    except Exception as e:
        print(f"Database error, falling back to file: {e}")
        # Fall back to file-based auth if database fails
        users = load_users()
        if username in users:
            if users[username]["password_hash"] == hash_password(password):
                return True
    
    return False

def register_user(username, email, password):
    """Register a new user in database and file (for backup)"""
    # Check if user exists in database
    user = get_user_by_username(username)
    if user:
        return False
    
    # Add to database
    try:
        password_hash = hash_password(password)
        add_user(username, email, password_hash)
        
        # Also update the JSON file as backup
        users = load_users()
        users[username] = {
            "password_hash": password_hash,
            "email": email,
            "created_at": time.time()
        }
        save_users(users)
        
        return True
    except Exception as e:
        print(f"Error registering user: {e}")
        return False

def login_user(username, password):
    """Log in a user and set session state"""
    if verify_user(username, password):
        st.session_state.authenticated = True
        st.session_state.username = username
        
        # Get user from database to store user_id in session
        try:
            user = get_user_by_username(username)
            if user:
                st.session_state.user_id = user.id
        except Exception as e:
            print(f"Note: Could not get user ID: {e}")
        
        return True
    return False

def logout_user():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.username = ""
    if "user_id" in st.session_state:
        del st.session_state.user_id

def require_auth():
    """Check if user is authenticated, redirect to login if not"""
    if not st.session_state.get("authenticated", False):
        st.switch_page("app.py")

# Initialize the user database on module load
initialize_users_file()
