from sklearn.linear_model import LogisticRegression
import numpy as np

def create_model():
    model = LogisticRegression(C=0.01, class_weight='balanced')

    model.classes_ = np.arange(13)      
    model.coef_ = np.zeros((2, 13)) 
    model.intercept_ = np.zeros(2)   
