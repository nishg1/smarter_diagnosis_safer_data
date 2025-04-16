import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import json
import flwr as fl

class HeartDiseaseModel:
    def __init__(self, csv_path=None):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # If a CSV path is provided, initialize with that data
        if csv_path:
            self.initialize_with_data(csv_path)

    def initialize_with_data(self, csv_path):
        # Load and preprocess data
        data = pd.read_csv(csv_path)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Fit scaler and train model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        return self

    def train(self, csv_path):
        # Load and preprocess data
        data = pd.read_csv(csv_path)
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return self
    
    def evaluate(self, test_csv_path):
        # Load and preprocess test data
        test_data = pd.read_csv(test_csv_path)
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100
        return accuracy

    def get_weights(self):
        # For Random Forest, we'll serialize the model parameters
        model_params = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'feature_importances': self.model.feature_importances_.tolist(),
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'classes': self.model.classes_.tolist()
        }
        return model_params

    def set_weights(self, weights):
        # Create a new model with the given parameters
        self.model = RandomForestClassifier(
            n_estimators=weights['n_estimators'],
            max_depth=weights['max_depth'],
            random_state=42
        )
        # Note: We can't directly set feature importances, so we'll just update the parameters
        return self
    
    def save(self, path):
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        return joblib.load(path)

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id, csv_path):
        self.client_id = client_id
        self.model = HeartDiseaseModel(csv_path)
        self.X_train, self.y_train = self.load_client_data(csv_path)

    def load_client_data(self, csv_path):
        data = pd.read_csv(csv_path)
        X = data.drop('target', axis=1)
        y = data['target']
        return X, y

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train(self.csv_path)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = self.model.evaluate(self.csv_path)
        return float(1 - acc), len(self.X_train), {"accuracy": acc} 