#! /usr/env/python

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

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

model = LGBMRegressor()#n_estimators=1000, learning_rate=0.01)
param_grid = {
    'learning_rate': [0.005, 0.01, 0.05, 0.07, 0.08, 0.09, 0.1],
    'n_estimators': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000],
    'boosting': ['gbdt', 'rf', 'dart', 'goss'],
    'min_data_in_leaf': [10, 20, 40, 60]
}
gbm = GridSearchCV(model, param_grid)
gbm.fit(X,Y, sample_weight = weights)

print('Best parameters found by grid search are:', gbm.best_params_)
print('Best score:', gbm.best_score_)
print('Feature importances:', list(gbm.best_estimator_.feature_importances_))

yp = pd.Series(gbm.predict(df_test[features])).rename('y')
yp.index.name = 'Index'
print(yp.head())

yp.to_csv('GradientBoostedRegressor7.csv', header=True)


# TODO: After this search is finished, also optimize over the number of leaves... maybe more leaves give better regressions
