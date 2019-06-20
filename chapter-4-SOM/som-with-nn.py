#! /usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show


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
frauds = np.concatenate((mappings[(7, 6)], mappings[(4, 4)], mappings[(4, 6)]), axis=0)

frauds = sc.inverse_transform(frauds)