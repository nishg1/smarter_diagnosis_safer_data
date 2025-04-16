from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from model import HeartDiseaseModel
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize the global model with test data
global_model = HeartDiseaseModel('data/test_heart_disease.csv')
client_weights = []

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    weights = request.json['weights']
    client_weights.append(weights)
    return jsonify({"status": "success"})

def aggregate_weights():
    if not client_weights:
        return None
    
    # For Random Forest, we'll average the feature importances
    avg_weights = {
        'n_estimators': client_weights[0]['n_estimators'],
        'max_depth': client_weights[0]['max_depth'],
        'feature_importances': np.mean([w['feature_importances'] for w in client_weights], axis=0).tolist(),
        'n_features': client_weights[0]['n_features'],
        'n_classes': client_weights[0]['n_classes'],
        'classes': client_weights[0]['classes']
    }
    
    return avg_weights

@app.route('/get_global_model', methods=['GET'])
def get_global_model():
    global global_model
    new_weights = aggregate_weights()
    if new_weights:
        # Create a new model with the test data and set the aggregated weights
        global_model = HeartDiseaseModel('data/test_heart_disease.csv')
        global_model.set_weights(new_weights)
        # Retrain the model with the test data
        global_model.train('data/test_heart_disease.csv')
    return jsonify({"status": "success"})

@app.route('/evaluate_global_model', methods=['GET'])
def evaluate_global():
    accuracy = global_model.evaluate('data/test_heart_disease.csv')
    return jsonify({"accuracy": accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file_path = request.json['file_path']
        data = pd.read_csv(file_path)
        
        # Scale the features
        X_scaled = global_model.scaler.transform(data)
        
        # Make predictions
        predictions = global_model.model.predict(X_scaled).tolist()
        
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 