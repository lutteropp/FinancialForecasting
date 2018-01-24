#! /usr/env/python

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6', 'pastY1', 'pastYDist1', 'futureY1', 'futureYDist1', 'weightedAvg1', 'semi_weightedAvg1', 'unweightedAvg1', 'pastY2', 'pastYDist2', 'futureY2', 'futureYDist2', 'weightedAvg2', 'semi_weightedAvg2', 'unweightedAvg2', 'pastY3', 'pastYDist3', 'futureY3', 'futureYDist3', 'weightedAvg3', 'semi_weightedAvg3', 'unweightedAvg3', 'pastY4', 'pastYDist4', 'futureY4', 'futureYDist4', 'weightedAvg4', 'semi_weightedAvg4', 'unweightedAvg4', 'pastY5', 'pastYDist5', 'futureY5', 'futureYDist5', 'weightedAvg5', 'semi_weightedAvg5', 'unweightedAvg5', 'pastY6', 'pastYDist6', 'futureY6', 'futureYDist6', 'weightedAvg6', 'semi_weightedAvg6', 'unweightedAvg6', 'pastY7', 'pastYDist7', 'futureY7', 'futureYDist7', 'weightedAvg7', 'semi_weightedAvg7', 'unweightedAvg7', 'pastY8', 'pastYDist8', 'futureY8', 'futureYDist8', 'weightedAvg8', 'semi_weightedAvg8', 'unweightedAvg8', 'pastY9', 'pastYDist9', 'futureY9', 'futureYDist9', 'weightedAvg9', 'semi_weightedAvg9', 'unweightedAvg9', 'pastY10', 'pastYDist10', 'futureY10', 'futureYDist10', 'weightedAvg10', 'semi_weightedAvg10', 'unweightedAvg10']

df = pd.read_csv('train_augmented.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
df_test = pd.read_csv('test_augmented.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries

weights = []
for index, row in df.iterrows():
	weights.append(float(row['Weight']))


X = df[features]
Y = df['y']

model = LGBMRegressor(n_estimators=10000, learning_rate=0.01)

model.fit(X,Y, sample_weight = weights)

yp = pd.Series(model.predict(df_test[features])).rename('y')
yp.index.name = 'Index'
print(yp.head())

yp.to_csv('GradientBoostedRegressorWithMoreFeatures.csv', header=True)

print(model.feature_importances_)
