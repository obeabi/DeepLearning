# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:39:27 2020

@author: obemb
"""


# Creating my first RNN 
## Written by Abiola Obembe
### Date:  2020-10-08

# Part 1- Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = dataset_train.iloc[:,1:2].values

# Feature scaling by normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_set_sc = sc.fit_transform(train_set)

# Create data structure using history (Tx) and 1 output
Tx = 60
X_train = []
y_train = []
m = len(train_set_sc)

for i in range(Tx, m):
    X_train.append(train_set_sc[i-Tx : i, 0])
    y_train.append(train_set_sc[i,0])
    
X_train, y_train = np.array(X_train) , np.array(y_train)

# Reshape to RNN Input shape including adding new indicators

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#%%
# Part 2 - Building the RNN

# Import the libraries from keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Initalize the RNN/LSTM

rnn = Sequential()

# Add the first layer and dropout regularization
rnn.add(LSTM(units = 50, return_sequences=True, input_shape=( X_train.shape[1],1)))
rnn.add(Dropout((0.2)))

# Add the Second layer and dropout regularization
rnn.add(LSTM(units = 50, return_sequences=True))
rnn.add(Dropout((0.2)))

# Add the Third layer and dropout regularization
rnn.add(LSTM(units = 50, return_sequences=True))
rnn.add(Dropout((0.2)))

# Add the Fourth layer and dropout regularization
rnn.add(LSTM(units = 50, return_sequences=False))
rnn.add(Dropout((0.2)))

# Add the output layer
rnn.add(Dense(units = 1))

# Compile the RNN
rnn.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the rnn to the training set
rnn.fit(X_train, y_train, epochs = 100, batch_size=32)

#%%

# Part 3 : Making the predictions and visualizing the results

# Get the real stock proce from the test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# Create input considering previous timesteps (60 i.e. Tx)
inputs = dataset_total[len(dataset_total)- len(dataset_test) - Tx :]. values

# reshape input
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

m_test = dataset_test.shape[0]

for i in range(Tx, Tx + m_test):
    X_test.append(inputs[i-Tx : i, 0])
  
    
X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%%
# Visualising the result
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title("Stock Price Prediction")
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()




