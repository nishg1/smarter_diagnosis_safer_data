from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def create_model():
    model = LogisticRegression()

    model.classes_ = np.arange(13)                      # e.g., [0, 1]
    model.coef_ = np.zeros((13))  # shape: (n_classes, n_features)
    model.intercept_ = np.zeros(n_classes if n_classes > 2 else 1)   
