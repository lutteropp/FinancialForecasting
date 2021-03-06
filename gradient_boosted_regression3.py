#! /usr/env/python

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

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

#model = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)
model = LGBMRegressor(n_estimators=1000, learning_rate=0.01, boosting_type = 'dart')

model.fit(X,Y, sample_weight = weights)
yp = pd.Series(model.predict(df_test[features])).rename('y')
yp.index.name = 'Index'
print(yp.head())

yp.to_csv('GradientBoostedRegressor3.csv', header=True)
