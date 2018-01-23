#! /usr/env/python

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from lightgbm import LGBMRegressor

features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']

df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
weights = []
for index, row in df.iterrows():
	weights.append(float(row['Weight']))
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries

X = df[features]
Y = df['y']

scaler = MinMaxScaler(feature_range=(0, 1))
model = MLPRegressor()

#param_grid = {
#    'max_iter': [1000]
#}
#gbm = GridSearchCV(model, param_grid)

#gbm.fit(scaler.fit_transform(X),Y, sample_weight=weights)

model.fit(scaler.fit_transform(X),Y)

#print('Best parameters found by grid search are:', gbm.best_params_)
#print('Best score:', gbm.best_score_)
#print('Feature importances:', list(gbm.best_estimator_.feature_importances_))

yp = pd.Series(model.predict(scaler.fit_transform(df_test[features]))).rename('y')
yp.index.name = 'Index'
print(yp.head())

yp.to_csv('MLPRegressor.csv', header=True)
