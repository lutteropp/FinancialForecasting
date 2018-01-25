#! /usr/env/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import math
import copy

def my_mse(ytrue, ypred, weights):
	res = 0.0
	for i in range(len(ytrue)):
		res = res + weights[i] * (ytrue[i] - ypred[i]) * (ytrue[i] - ypred[i])
	return res

class MyPredictor:
	def fit(self, X, Y):
		# got this from manually looking at the training data
		minDay = 1
		maxDay = 729
		minStock = 0
		maxStock = 3022

		y_vals = [[0.0 for x in range(maxDay+1)] for y in range(maxStock+1)]
		missing = [[True for x in range(maxDay+1)] for y in range(maxStock+1)]
		for index, row in X.iterrows():
			stock = int(row['Stock'])
			day = int(row['Day'])
			y = float(Y[index])
			y_vals[stock][day] = y
			missing[stock][day] = False

		# do forward ANO
		self.y_vals_forward = copy.deepcopy(y_vals)
		missing_forward = copy.deepcopy(missing)
		for i in range(maxStock + 1):
			for j in range(maxDay + 1):
				if missing_forward[i][j]:
					pastVal = 0
					pastValFound = False
					pastDay = j-1
					while pastDay >= minDay:
						if not missing_forward[i][pastDay]:
							pastVal = self.y_vals_forward[i][pastDay]
							pastValFound = True
							break
						pastDay = pastDay - 1
					futureVal = 0
					futureValFound = False
					futureDay = j+1
					while futureDay <= maxDay:
						if not missing_forward[i][futureDay]:
							futureVal = self.y_vals_forward[i][futureDay]
							futureValFound = True
							break
						futureDay = futureDay + 1
					if pastValFound and futureValFound:
						self.y_vals_forward[i][j] = (pastVal + futureVal) / 2.0
					elif pastValFound:
						self.y_vals_forward[i][j] = pastVal
					elif futureValFound:
						self.y_vals_forward[i][j] = futureVal
					else:
						self.y_vals_forward[i][j] = 0.0
					missing_forward[i][j] = False

		# do backward ANO
		self.y_vals_backward = copy.deepcopy(y_vals)
		missing_backward = copy.deepcopy(missing)
		for i in range(maxStock + 1):
			for j in reversed(range(maxDay + 1)):
				if missing_backward[i][j]:
					pastVal = 0
					pastValFound = False
					pastDay = j-1
					while pastDay >= minDay:
						if not missing_backward[i][pastDay]:
							pastVal = self.y_vals_backward[i][pastDay]
							pastValFound = True
							break
						pastDay = pastDay - 1
					futureVal = 0
					futureValFound = False
					futureDay = j+1
					while futureDay <= maxDay:
						if not missing_backward[i][futureDay]:
							futureVal = self.y_vals_backward[i][futureDay]
							futureValFound = True
							break
						futureDay = futureDay + 1
					if pastValFound and futureValFound:
						self.y_vals_backward[i][j] = (pastVal + futureVal) / 2.0
					elif pastValFound:
						self.y_vals_backward[i][j] = pastVal
					elif futureValFound:
						self.y_vals_backward[i][j] = futureVal
					else:
						self.y_vals_backward[i][j] = 0.0
					missing_backward[i][j] = False

	def predict_forward_ano(self, X):
		res = []
		for index, row in X.iterrows():
			res.append(self.y_vals_forward[int(row['Stock'])][int(row['Day'])])
		return res

	def predict_backward_ano(self, X):
		res = []
		for index, row in X.iterrows():
			res.append(self.y_vals_backward[int(row['Stock'])][int(row['Day'])])
		return res

	def predict_twodir_ano(self, X):
		res = []
		for index, row in X.iterrows():
			res.append((self.y_vals_forward[int(row['Stock'])][int(row['Day'])] + self.y_vals_backward[int(row['Stock'])][int(row['Day'])]) / 2.0)
		return res

	def score(self, y_true, y_pred, weights):
		return my_mse(list(y_true), list(y_pred), list(weights))



print("Reading data...")
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
print("Splitting dataset...")
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, Y, weights, test_size=0.5, random_state=42)
print("Fitting predictor...")
mypred = MyPredictor()
mypred.fit(X_train,y_train)

y_pred_forward = mypred.predict_forward_ano(df_test)
print("Predicting with forward ANO score: " + str(mypred.score(y_test, y_pred_forward, sw_test)))
y_pred_backward = mypred.predict_backward_ano(df_test)
print("Predicting with backward ANO score: " + str(mypred.score(y_test, y_pred_backward, sw_test)))
y_pred_twodir = mypred.predict_twodir_ano(df_test)
print("Predicting with twodir ANO score: " + str(mypred.score(y_test, y_pred_twodir, sw_test)))
