#! /usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


dataset = pd.read_csv("chapter-4-SOM/dataset/Credit_Card_Applications.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Scale Change
sc = MinMaxScaler()
X = sc.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualization
bone()
pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
colors = ["r", "g"]
for idx, x in enumerate(X):
    w = som.winner(x)
    plot(
        w[0] + 0.5, w[1] + 0.5, markers[y[idx]],
        markeredgecolor=colors[y[idx]],
        markerfacecolor="None",
        markersize=10,
        markeredgewidth=2)

show()

# Fraud Detect
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1, 4)], mappings[(5, 8)]), axis=0)

frauds = sc.inverse_transform(frauds)


# Neural Network Part

# make matrix variables
customers = dataset.iloc[:, 1:].values

# make dependent variable
is_fraud = np.zeros(len(dataset))
for idx in range(0, len(dataset)):
    if dataset.iloc[idx, 0] in frauds:
        is_fraud[idx] = 1

# Encoding Categorical Data and Feature Scaling
preprocess = StandardScaler()
customers = preprocess.fit_transform(customers)

# Initialization Neuronal Network
classifier = Sequential()

# Add Hidden Layers
classifier.add(Dense(units=2, activation="relu", kernel_initializer="uniform", input_dim=8))

# Add Output Layer
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# compile Neural Network
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training Neural Network
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Prediction with Test set
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1], y_pred), axis=1)

y_pred = y_pred[y_pred[:, 1].argsort()]