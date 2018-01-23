#! /usr/env/python

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']

df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries

X = df[features]
Y = df['y']

model = AdaBoostRegressor(DecisionTreeRegressor())
#model = LGBMRegressor(n_estimators=1000, learning_rate=0.01)

model.fit(X,Y)
yp = pd.Series(model.predict(df_test[features])).rename('y')
yp.index.name = 'Index'
print(yp.head())

yp.to_csv('AdaBoostRegressor.csv', header=True)
