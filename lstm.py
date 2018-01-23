#! /usr/env/python

import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries
weights = []
for index, row in df.iterrows():
	weights.append(float(row['Weight']))
X = df[features]
Y = df['y']

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))

model = Sequential()
model.add(LSTM(4, input_shape=(1, len(features))))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(scaler.fit_transform(X), scaler.fit_transform(Y), sample_weight = weights, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(scaler.fit_transform(df_test[features]))

yp = pd.Series(scaler.inverse_transform(model.predict(df_test[features]))).rename('y')
yp.index.name = 'Index'
print(yp.head())
yp.to_csv('lstm.csv', header=True)
