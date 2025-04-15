from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

def create_model():
    model = HistGradientBoostingClassifier()
    # model.classes_ = np.array([0, 1])
    # model.coef_ = np.zeros((1, 3))
    # model.intercept_ = np.zeros(1)
