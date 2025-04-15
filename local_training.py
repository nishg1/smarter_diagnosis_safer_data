import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ucimlrepo import fetch_ucirepo



def load_client_data(client_id):
    path = f"hospital_{client_id}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data found for client {client_id} at {path}")

    df = pd.read_csv(path)

    X = df.drop(columns=["target"]).values
    y = df["target"].values

    # # fetch dataset
    # heart_disease = fetch_ucirepo(id=45)

    # X = heart_disease.data.features
    # y = heart_disease.data.targets

    # y = y['num'].apply(lambda x: 1 if x > 0 else 0) # convert to either 0 for no heart disease or 1 for heart disease

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
