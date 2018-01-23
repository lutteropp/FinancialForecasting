#! /usr/env/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

import math

def update_future(actDay, act_window_idx, max_window_idx, maxDay, filledDays, yVals, futureYVals, futureYDist):
	nextDay = actDay + 1
	while nextDay <= maxDay:
		if nextDay in filledDays[row['Stock']]:
			futureYVals[act_window_idx][index] = yVals[(row['Stock'], nextDay)]
			futureYDist[act_window_idx][index] = nextDay - actDay
			if (act_window_idx < max_window_idx):
				update_future(nextDay + 1, act_window_idx + 1, max_window_idx, maxDay, filledDays, yVals, futureYVals, futureYDist)
			break
		nextDay = nextDay + 1

def update_past(actDay, act_window_idx, max_window_idx, minDay, filledDays, yVals, pastYVals, pastYDist):
	lastDay = actDay - 1
	while lastDay >= minDay:
		if lastDay in filledDays[row['Stock']]:
			pastYVals[act_window_idx][index] = yVals[(row['Stock'], lastDay)]
			pastYDist[act_window_idx][index] = actDay - lastDay
			if (act_window_idx < max_window_idx):
				update_past(lastDay - 1, act_window_idx + 1, max_window_idx, minDay, filledDays, yVals, pastYVals, pastYDist)
			break
		lastDay = lastDay - 1

def process_index(index, row, window, maxDay, filledDays, yVals, futureYVals, futureYDist, minDay, yVals, pastYVals, pastYDist, weightedAvg):
	print(index)
	actDay = int(row['Day'])

	for i in range(window):
		futureYDist[i][index] = 999
		pastYDist[i][index] = 999

	update_future(actDay, 0, window - 1, maxDay, filledDays, yVals, futureYVals, futureYDist)
	update_past(actDay, 0, window - 1, minDay, filledDays, yVals, pastYVals, pastYDist)

	sumValPast = 0
	sumValNext = 0
	sumWeightPast = 0
	sumWeightNext = 0
	for i in range(window):
		sumValPast = sumValPast + (1/float(pastYDist[i][index])) * pastYVals[i][index]
		sumValNext = sumValNext + (1/float(futureYDist[i][index])) * futureYVals[i][index]
		sumWeightPast = sumWeightPast + (1/ float(pastYDist[i][index]))
		sumWeightNext = sumWeightNext + (1 / float(futureYDist[i][index]))
		weightedAvg[i][index] = (sumValPast + sumValNext) / (sumWeightPast + sumWeightNext)

def augment_dataframe(window, dataframe, minDay, maxDay, filledDays, yVals, orig_features):
	num_entries = 0
	for index, row in dataframe.iterrows():
		num_entries = num_entries + 1

	pastYVals = []
	futureYVals = []
	pastYDist = []
	futureYDist = []
	weightedAvg = []
	for i in range(window):
		pastYVals.append(np.zeros(num_entries))
		futureYVals.append(np.zeros(num_entries))
		pastYDist.append(np.zeros(num_entries))
		futureYDist.append(np.zeros(num_entries))
		weightedAvg.append(np.zeros(num_entries))

	Parallel(n_jobs=4)(delayed(process_index)(index, row, window, maxDay, filledDays, yVals, futureYVals, futureYDist, minDay, yVals, pastYVals, pastYDist, weightedAvg) for index, row in dataframe.iterrows())

	#for index, row in dataframe.iterrows():
	#	process_index(index, row, window, maxDay, filledDays, yVals, futureYVals, futureYDist, minDay, yVals, pastYVals, pastYDist, weightedAvg)

	features = copy(orig_features)
	for i in range(window):
		dataframe['pastY' + str(i+1)]=pastYVals[i]
		dataframe['pastYDist' + str(i+1)]=pastYDist[i]
		dataframe['futureY' + str(i+1)]=futureYVals[i]
		dataframe['futureYDist' + str(i+1)]=futureYDist[i]
		dataframe['weightedAvg' + str(i+1)] = weightedAvg[i]

		features.append('pastY' + str(i+1))
		features.append('pastYDist' + str(i+1))
		features.append('futureY' + str(i+1))
		features.append('futureYDist' + str(i+1))
		features.append('weightedAvg' + str(i+1))
	return features


print("Reading data...")
df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries
num_entries = 0

yVals = {}
filledDays = {}
print("Part 1...")
minDay = 999999
maxDay = 0
for index, row in df.iterrows():
	num_entries = num_entries + 1
	yVals[(row['Stock'], int(row['Day']))] = float(row['y'])
	if row['Stock'] not in filledDays:
		filledDays[row['Stock']] = set()
	filledDays[row['Stock']].add(int(row['Day']))	
	if int(row['Day']) < minDay:
		minDay = int(row['Day'])
	if int(row['Day']) > maxDay:
		maxDay = int(row['Day'])

window = 4

orig_features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
print("Part 2...")
aug_features = augment_dataframe(window, df, minDay, maxDay, filledDays, yVals, orig_features)
df.to_csv("train_augmented.csv")
print("Part 3...")

aug_features_plus_y = ['y']
for f in aug_features:
	aug_features_plus_y.append(f)

g = sns.heatmap(df[aug_features_plus_y].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()
print("Part 4...")

for index, row in df_test.iterrows():
	if row['Stock'] not in filledDays:
		filledDays[row['Stock']] = set()
		print("Stock number " + row['Stock'] + " does not appear in the training data.")

augment_dataframe(window, df_test, minDay, maxDay, filledDays, yVals, orig_features)
df_test.to_csv("test_augmented.csv")
