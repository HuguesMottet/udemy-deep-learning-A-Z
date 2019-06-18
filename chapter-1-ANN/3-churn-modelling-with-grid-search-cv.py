#! /usr/bin/env python3

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
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
def build_model(optimizer="adam"):
    network = Sequential()

    # Add Hidden Layers and Add Dropout
    network.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
    network.add(Dropout(rate=0.1))
    network.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    network.add(Dropout(rate=0.1))

    # Add Output Layer
    network.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

    # compile Neural Network
    network.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return network


classifier = KerasClassifier(build_fn=build_model)

parameters = {
    "batch_size": [10, 25, 32],
    "epochs": [100, 500, 800],
    "optimizer": ["adam", "rmsprop"],
}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv=10)

grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracies = grid_search.best_score_