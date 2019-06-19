#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load Data
dataset_train = pd.read_csv("chapter-3-RNN/dataset/Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values

# Feature Scaling
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# Structure Creation with 60 timesteps
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[(i - 60):i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Init rnn
regressor = Sequential()

# 1 layer
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 2 layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 3 layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 4 layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Out layer
regressor.add(Dense(units=1))

# Compilation
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Training
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict
dataset_test = pd.read_csv("chapter-3-RNN/dataset/Google_Stock_Price_Test.csv")
test_set = dataset_test[["Open"]].values

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs_scaled [(i - 60):i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

sc.inverse_transform(predicted_stock_price)

plt.plot(test_set, color="red", label="Prix reel de l'action Google")
plt.plot(predicted_stock_price, color="green", label="Prix predit de l'action Google")
plt.title("Prediction de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()