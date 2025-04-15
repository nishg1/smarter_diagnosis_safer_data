from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from model import create_model
import wandb
from datetime import datetime
from flask import Flask, request, jsonify
import threading
import time
from flwr.server import ServerConfig

# Initialize Flask app
app = Flask(__name__)

# Initialize Weights & Biases
wandb.login(key="0bfff8cb0cebdb5ac084581e40317fae5f12cc1a")

class SaveModelStrategy(FedAvg):
    def __init__(self):
        super().__init__()
        self.latest_parameters = None
        self.client_metrics = {}  # Store metrics from each client
        self.client_parameters = {}  # Store parameters from each client

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result is not None:
            self.latest_parameters = aggregated_result[0]  # model parameters
            
            # Log metrics to Weights & Biases
            for client_id, metrics in self.client_metrics.items():
                wandb.log({
                    "round": rnd,
                    "client_id": client_id,
                    "timestamp": datetime.now().isoformat(),
                    **metrics
                })
            
            self.client_metrics = {}  # Clear metrics after logging
        return aggregated_result

    def configure_evaluate(self, rnd, parameters, client_manager):
        # Store client metrics during evaluation
        def evaluate_config(rnd, client_id):
            return {"client_id": client_id}
        
        return evaluate_config

    def aggregate_evaluate(self, rnd, results, failures):
        # Store metrics from each client
        for result in results:
            client_id = result[1]["client_id"]
            metrics = result[1]["metrics"]
            self.client_metrics[client_id] = metrics
        
        return super().aggregate_evaluate(rnd, results, failures)

def set_model_params(model, parameters):
    model.coef_ = np.array(parameters[0])
    model.intercept_ = np.array(parameters[1])
    model.classes_ = np.array([0, 1])
    return model

@app.route('/update_parameters', methods=['POST'])
def update_parameters():
    data = request.json
    client_id = data['client_id']
    username = data['username']
    parameters = [np.array(p) for p in data['parameters']]
    
    # Store parameters
    strategy.client_parameters[client_id] = {
        'parameters': parameters,
        'username': username,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify({"status": "success"})

def start_flask_server():
    app.run(host='0.0.0.0', port=8080)

def start_server():
    global strategy
    strategy = SaveModelStrategy()
    
    # Initialize Weights & Biases run
    wandb.init(
        project="federated_learning_heart_disease",
        name=f"server_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "strategy": "FedAvg",
            "num_rounds": 5
        }
    )
    
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True  # This ensures the thread will be killed when the main program exits
    flask_thread.start()
    
    # Give Flask server time to start
    time.sleep(2)
    
    # Start Flower server in the main thread
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=5),
        strategy=strategy
    )

def start_federated_client(user_id, username):
    client = HospitalClient(user_id, username)

if __name__ == "__main__":
    start_server() 