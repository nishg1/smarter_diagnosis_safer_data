import streamlit as st
import pandas as pd
import os
from datetime import datetime
import requests
from model import HeartDiseaseModel
from database import (
    create_user, verify_user, get_user_profile, update_profile,
    log_action, get_user_actions
)

# Set page configuration
st.set_page_config(
    page_title="Federated Learning for Heart Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'local_model' not in st.session_state:
    st.session_state.local_model = None

def login_page():
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user_id = verify_user(username, password)
            if user_id:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_id = user_id
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    st.write("Don't have an account?")
    if st.button("Register"):
        st.session_state.show_register = True
        st.experimental_rerun()

def register_page():
    st.title("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                if create_user(username, password, email):
                    st.success("Registration successful! Please login.")
                    st.session_state.show_register = False
                    st.experimental_rerun()
                else:
                    st.error("Username or email already exists!")

def train_local_model(uploaded_file):
    # Save the uploaded file
    file_path = os.path.join('uploads', f"{st.session_state.user_id}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Log the document upload
    log_action(
        st.session_state.user_id,
        "document_upload",
        {"filename": uploaded_file.name, "type": uploaded_file.type}
    )
    
    # Initialize and train model
    model = HeartDiseaseModel()
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Train model
    model.train(file_path)
    progress_bar.progress(100)
    status_text.text("Training completed!")
    
    st.session_state.local_model = model
    
    # Evaluate local model
    accuracy = model.evaluate('data/test_heart_disease.csv')
    st.success(f"Local model trained successfully! Test accuracy: {accuracy:.2f}%")
    
    # Log the training event
    log_action(
        st.session_state.user_id,
        "model_training",
        {"accuracy": accuracy, "filename": uploaded_file.name}
    )
    
    # Submit weights to server
    weights = model.get_weights()
    
    try:
        response = requests.post('http://localhost:5000/submit_weights', 
                               json={'weights': weights})
        if response.status_code == 200:
            st.success("Model weights submitted to server successfully!")
        else:
            st.error("Failed to submit weights to server")
    except Exception as e:
        st.error(f"Error connecting to server: {str(e)}")

def document_upload_page():
    st.title("Document Upload and Training")
    
    if not st.session_state.authenticated:
        st.warning("Please login first to upload documents.")
        return
    
    st.write("""
    ### Upload your dataset
    Upload your data file to train a local model. The system will automatically process your data and train a model.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        if st.button("Train Model"):
            train_local_model(uploaded_file)

def inference_page():
    st.title("Model Inference")
    
    if not st.session_state.authenticated:
        st.warning("Please login first to use the inference service.")
        return
    
    st.write("""
    ### Upload data for prediction
    Upload your data file to get predictions from the global model.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    
    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join('uploads', f"{st.session_state.user_id}_inference_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Get predictions from the global model
            response = requests.post('http://localhost:5000/predict', 
                                   json={'file_path': file_path})
            
            if response.status_code == 200:
                predictions = response.json()['predictions']
                
                # Display predictions
                st.subheader("Predictions")
                data = pd.read_csv(file_path)
                data['Predicted Target'] = predictions
                st.dataframe(data)
                
                # Log the inference event
                log_action(
                    st.session_state.user_id,
                    "inference",
                    {"filename": uploaded_file.name, "num_predictions": len(predictions)}
                )
            else:
                st.error("Failed to get predictions from the server")
        except Exception as e:
            st.error(f"Error connecting to server: {str(e)}")

def profile_page():
    st.title("User Profile")
    
    if st.session_state.username:
        profile = get_user_profile(st.session_state.username)
        if profile:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Profile Information")
                st.write(f"Username: {profile['username']}")
                st.write(f"Email: {profile['email']}")
                st.write(f"Member since: {profile['created_at']}")
                
                # Profile data editor
                profile_data = st.text_area("Additional Information", value=profile['profile_data'] or "")
                if st.button("Save Profile"):
                    update_profile(st.session_state.username, profile_data)
                    st.success("Profile updated successfully!")
            
            with col2:
                st.subheader("Action History")
                actions = get_user_actions(st.session_state.user_id)
                for action in actions:
                    st.write(f"**{action['type']}** - {action['date']}")
                    if action['details']:
                        st.write(f"Details: {action['details']}")
                    st.write("---")

def main_app():
    st.sidebar.title("Navigation")
    
    if st.session_state.authenticated:
        st.sidebar.write(f"Welcome, {st.session_state.username}!")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_id = None
            st.experimental_rerun()
        
        page = st.sidebar.radio("Go to", ["Profile", "Document Upload", "Inference"])
        
        if page == "Profile":
            profile_page()
        elif page == "Document Upload":
            document_upload_page()
        elif page == "Inference":
            inference_page()
    else:
        st.sidebar.write("Please login to access the app")

# Main app logic
if not st.session_state.authenticated:
    if hasattr(st.session_state, 'show_register') and st.session_state.show_register:
        register_page()
    else:
        login_page()
else:
    main_app() 