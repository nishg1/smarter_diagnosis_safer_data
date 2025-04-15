import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import wandb
from database import (
    create_user, verify_user, get_user_profile, update_profile,
    log_action, get_user_actions, save_document, get_user_documents
)
from local_training import load_client_data, train_model, evaluate_model
from model import create_model
import flwr as fl
import threading
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Medical Analysis App",
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

def profile_page():
    st.title("Profile")
    
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

def document_upload_page():
    st.title("Document Upload")
    
    if not st.session_state.authenticated:
        st.warning("Please login first to upload documents.")
        return
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json'])
    
    if uploaded_file is not None:
        # Save the file
        file_path = os.path.join('uploads', f"{st.session_state.user_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Save document info to database
        save_document(
            st.session_state.user_id,
            uploaded_file.name,
            uploaded_file.type,
            file_path
        )
        
        # Initialize Weights & Biases for this client
        wandb.login(key="0bfff8cb0cebdb5ac084581e40317fae5f12cc1a")
        wandb.init(
            project="federated_learning_heart_disease",
            name=f"client_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "client_id": st.session_state.user_id,
                "username": st.session_state.username
            }
        )
        
        # Create a progress bar in the UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Store user info and file path in variables to pass to thread
        user_id = st.session_state.user_id
        username = st.session_state.username
        
        # Start federated learning client in a separate thread
        def start_federated_client(user_id, username, file_path):
            class HospitalClient(fl.client.NumPyClient):
                def __init__(self, client_id, username, file_path):
                    self.client_id = client_id
                    self.username = username
                    self.model = create_model()
                    self.X_train, self.X_test, self.y_train, self.y_test = load_client_data(file_path)
                    self.progress = 0
                
                def get_parameters(self, config):
                    # For HistGradientBoostingClassifier, we need to get the model's state
                    # We'll use the model's state_dict() method if available, or create a custom state
                    try:
                        # Try to get the model's state
                        state = self.model.get_params()
                        return [state]
                    except AttributeError:
                        # If get_params() is not available, return a simple state
                        return [{"model_type": "HistGradientBoostingClassifier"}]
                
                def set_parameters(self, parameters):
                    # For HistGradientBoostingClassifier, we need to set the model's state
                    # We'll use the model's set_params() method if available
                    try:
                        self.model.set_params(**parameters[0])
                    except AttributeError:
                        # If set_params() is not available, create a new model
                        self.model = create_model()
                
                def fit(self, parameters, config):
                    self.set_parameters(parameters)
                    
                    # Update progress
                    self.progress = 0
                    progress_bar.progress(self.progress)
                    status_text.text("Starting local training...")
                    
                    # Train model with progress tracking
                    for epoch in tqdm(range(100), desc="Training Progress"):
                        self.model = train_model(self.model, self.X_train, self.y_train)
                        self.progress = (epoch + 1) / 100
                        progress_bar.progress(self.progress)
                        status_text.text(f"Training epoch {epoch + 1}/100")
                    
                    # Calculate all metrics
                    y_pred = self.model.predict(self.X_train)
                    y_pred_proba = self.model.predict_proba(self.X_train)
                    
                    metrics = {
                        "training_accuracy": accuracy_score(self.y_train, y_pred),
                        "training_precision": precision_score(self.y_train, y_pred),
                        "training_recall": recall_score(self.y_train, y_pred),
                        "training_f1": f1_score(self.y_train, y_pred),
                        "training_roc_auc": roc_auc_score(self.y_train, y_pred_proba[:, 1]),
                        "training_log_loss": log_loss(self.y_train, y_pred_proba),
                        "client_id": self.client_id,
                        "username": self.username,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Log metrics to Weights & Biases
                    wandb.log(metrics)
                    
                    # Send parameters to server through API
                    params = self.get_parameters(config)
                    try:
                        response = requests.post(
                            "http://localhost:8080/update_parameters",
                            json={
                                "client_id": self.client_id,
                                "username": self.username,
                                "parameters": params
                            }
                        )
                        if response.status_code != 200:
                            st.error("Failed to send parameters to server")
                    except Exception as e:
                        st.error(f"Error sending parameters to server: {str(e)}")
                    
                    return params, len(self.X_train), {}
                
                def evaluate(self, parameters, config):
                    self.set_parameters(parameters)
                    
                    # Calculate all metrics
                    y_pred = self.model.predict(self.X_test)
                    y_pred_proba = self.model.predict_proba(self.X_test)
                    
                    metrics = {
                        "test_accuracy": accuracy_score(self.y_test, y_pred),
                        "test_precision": precision_score(self.y_test, y_pred),
                        "test_recall": recall_score(self.y_test, y_pred),
                        "test_f1": f1_score(self.y_test, y_pred),
                        "test_roc_auc": roc_auc_score(self.y_test, y_pred_proba[:, 1]),
                        "test_log_loss": log_loss(self.y_test, y_pred_proba),
                        "client_id": self.client_id,
                        "username": self.username,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Log metrics to Weights & Biases
                    wandb.log(metrics)
                    
                    return float(1 - metrics["test_accuracy"]), len(self.X_test), {
                        "accuracy": metrics["test_accuracy"],
                        "client_id": self.client_id,
                        "metrics": metrics
                    }
            
            client = HospitalClient(user_id, username, file_path)
            fl.client.start_numpy_client(server_address="localhost:8080", client=client)
            wandb.finish()
        
        # Start the client in a separate thread with user info and file path
        client_thread = threading.Thread(target=start_federated_client, args=(user_id, username, file_path))
        client_thread.start()
        
        # Log the action
        log_action(
            st.session_state.user_id,
            "document_upload",
            {"filename": uploaded_file.name, "type": uploaded_file.type}
        )
        
        st.success(f"File {uploaded_file.name} uploaded successfully! Training has started in the background.")
        
        # Show uploaded documents
        st.subheader("Your Uploaded Documents")
        documents = get_user_documents(st.session_state.user_id)
        for doc in documents:
            st.write(f"**{doc['filename']}** - {doc['upload_date']}")
            st.write(f"Type: {doc['type']}")
            st.write("---")

def inference_page():
    st.title("Inference")
    
    # File upload section
    st.subheader("Upload Data for Inference")
    uploaded_file = st.file_uploader("Choose a file for inference", type=['csv', 'xlsx', 'xls', 'json'])
    
    if uploaded_file is not None:
        # Save the file temporarily
        file_path = os.path.join('uploads', f"temp_{st.session_state.user_id}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display file preview
        if uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            st.write("File Preview:")
            st.dataframe(df.head())
        elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel']:
            df = pd.read_excel(uploaded_file)
            st.write("File Preview:")
            st.dataframe(df.head())
        
        # Inference options (to be implemented based on specific requirements)
        st.subheader("Inference Options")
        # Add your inference options here
        
        if st.button("Run Inference"):
            # Log the inference action
            log_action(
                st.session_state.user_id,
                "inference_run",
                {"filename": uploaded_file.name, "type": uploaded_file.type}
            )
            st.success("Inference completed successfully!")
            
            # Display results (to be implemented based on specific requirements)
            st.subheader("Results")
            st.write("Results will be displayed here")

def main_app():
    # Add a sidebar
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