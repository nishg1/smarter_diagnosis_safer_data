import flwr as fl
import numpy as np
from local_training import load_client_data, train_model, evaluate_model
from model import create_model

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = create_model()
        print("created client model")
        self.X_train, self.X_test, self.y_train, self.y_test = load_client_data(client_id)

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model = train_model(self.model, self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = evaluate_model(self.model, self.X_test, self.y_test)
        return float(1 - acc), len(self.X_test), {"accuracy": acc}

def start_client(client_id):
    fl.client.start_numpy_client(server_address="localhost:8080", client=HospitalClient(client_id))
