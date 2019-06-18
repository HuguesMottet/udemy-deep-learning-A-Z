#! /usr/bin/env python3

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense


# Global Variable
PATH_DATASET = "chapter-1-ANN/dataset/Churn_Modelling.csv"

# Import Dataset
dataset = pd.read_csv(PATH_DATASET)
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# Encoding Categorical Data and Feature Scaling
preprocess = ColumnTransformer([
    ("OneHotEncoding", OneHotEncoder(), [1, 2]),
    ("StandardScaler", StandardScaler(), [0, 3, 4, 5, 6, 7, 8, 9])
])
X = preprocess.fit_transform(X)
X = np.delete(X, [0, 3], 1)

# Splitting the Dataset in Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Initialization Neuronal Network
classifier = Sequential()

# Add Hidden Layers
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

# Add Output Layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# compile Neural Network
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training Neural Network
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Prediction with Test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting a single new observation
"""
Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""
Xnew = pd.DataFrame(data={
        'CreditScore': [600],
        'Geography': ['France'],
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0, 3], 1)
new_prediction = classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)