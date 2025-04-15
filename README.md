# federated_learning_hospital_data

There is untapped potential in being able to combine multiple hospitals' patient data in order to create a robust model trained on a very large set of data. A key concern is, understandably, patient privacy. To address this, our model uses federated learning, where a local model is trained on individual hospital's patient data. Subsequently, each set of parameters are sent to the global model. Sensitive patient data cannot be accessed by individual hospitals in our interface, while simultaneously, the user still benefits from having access to a huge model. 

Our interface shows how we can use many hospitals' patient data to predict heart disease. Heart disease continues to be one of the leading causes of death across all ethnic groups and people in various geographic locations. Given certain features, namely age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, and thal, we have curated a model that predicts whether or not a patient has heart disease. In the case that we predict that the patient does have heart disease, we recommend consulting a doctor for further steps. We use Logistic Regression as our model and have achieved a minimum accuracy of 88%. 


Team Members:
Mirelle George
Nishka Govil
Metehan Berker
