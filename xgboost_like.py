#! /usr/env/python

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

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

model = XGBRegressor()

model.fit(X,Y, sample_weight = weights)
yp = pd.Series(model.predict(df_test[features])).rename('y')
yp.index.name = 'Index'
print(yp.head())

yp.to_csv('XGBOOST.csv', header=True)
