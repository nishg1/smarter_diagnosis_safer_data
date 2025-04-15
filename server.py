from flwr.server.strategy import FedAvg
import flwr as fl
import numpy as np
from model import create_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.array([0, 1])
    return model


def main():
    strategy = SaveModelStrategy()

    print("entering into server")

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy)

    print("finished getting each client's parameters")

    global_params = strategy.latest_parameters
    model = create_model()
    global_params_array = fl.common.parameter.parameters_to_ndarrays(global_params)
    model = set_model_params(model, global_params_array)

    print("model coefficients:", model.coef_)
    print("model intercept:", model.intercept_)

    ## Change the following csv with the csv that the user inputs 
    ## Format of user-inputted test data: csv containing 1 row
    test_data = pd.read_csv("data/test_heart_disease_1.csv")
    row = test_data.iloc[0]
    y = row["target"]
    X = row.drop("target").values.reshape(1, -1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    y_pred = model.predict(X)

    if y_pred == 1:
        print("The patient is predicted to have heart disease. Please consult your healthcare provider")
    else:
        print("The patient is not predicted to have heart disease.")

if __name__ == "__main__":
    main()