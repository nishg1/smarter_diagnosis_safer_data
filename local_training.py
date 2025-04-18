import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer



def load_client_data(client_id):
    path = f"data/train_heart_disease_{client_id}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data found for client {client_id} at {path}")

    df = pd.read_csv(path)

    y = df["target"].values
    X = df.drop(columns=["target"]).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test) 

    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
