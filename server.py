from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from model import create_model

class SaveModelStrategy(FedAvg):
    def __init__(self):
        super().__init__()
        self.latest_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result is not None:
            self.latest_parameters = aggregated_result[0]  # model parameters
        return aggregated_result


def set_model_params(model, parameters):
    model.coef_ = np.array(parameters[0])
    model.intercept_ = np.array(parameters[1])
    model.classes_ = np.array([0, 1])  # Make sure this matches your task
    return model


def main():
    strategy = SaveModelStrategy()

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy)

    global_params = strategy.latest_parameters
    model = create_model()
    model = set_model_params(model, global_params)

    print(model.coef_)
    print(model.intercept_)

if __name__ == "__main__":
    main()