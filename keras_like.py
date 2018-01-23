#! /usr/env/python

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']

print("Reading data...")
df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries
print("Reading weights...")
weights_n = []
for index, row in df.iterrows():
	weights_n.append(float(row['Weight']))
weights = np.zeros(len(weights_n))
for i in range(len(weights_n)):
	weights[i] = weights_n[i]
X = df[features]
Y = df['y']

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(len(features), input_dim=len(features), kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print("Evaluating model...")

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

estimator.fit(X,Y, sample_weight = weights)
yp = pd.Series(estimator.predict(df_test[features])).rename('y')
yp.index.name = 'Index'
print(yp.head())
yp.to_csv('keras_like_simple.csv', header=True)

print("Doing 3x cross validation...")
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
