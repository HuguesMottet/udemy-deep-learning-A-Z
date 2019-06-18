#! /usr/bin/env python3

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

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
def build_model():
    network = Sequential()

    # Add Hidden Layers and Add Dropout
    network.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
    network.add(Dropout(rate=0.1))
    network.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    network.add(Dropout(rate=0.1))

    # Add Output Layer
    network.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

    # compile Neural Network
    network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return network


classifier = KerasClassifier(build_fn=build_model, batch_size=10, epochs=100)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

mean = accuracies.mean()
variance = accuracies.std()
