#! /usr/env/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import math

def update_future(actDay, actDay_orig, act_window_idx, max_window_idx, maxDay, filledDays, yVals, futureYVals, futureYDist, index, row, semi_futureYDist):
	nextDay = actDay + 1
	while nextDay <= maxDay:
		if nextDay in filledDays[row['Stock']]:
			futureYVals[act_window_idx][index] = yVals[(row['Stock'], nextDay)]
			futureYDist[act_window_idx][index] = nextDay - actDay_orig
			semi_futureYDist[act_window_idx][index] = nextDay - actDay
			if (act_window_idx < max_window_idx):
				return update_future(nextDay + 1, actDay_orig, act_window_idx + 1, max_window_idx, maxDay, filledDays, yVals, futureYVals, futureYDist, index, row, semi_futureYDist)
			break
		nextDay = nextDay + 1
	return (futureYVals, futureYDist, semi_futureYDist)

def update_past(actDay, actDay_orig, act_window_idx, max_window_idx, minDay, filledDays, yVals, pastYVals, pastYDist, index, row, semi_pastYDist):
	lastDay = actDay - 1
	while lastDay >= minDay:
		if lastDay in filledDays[row['Stock']]:
			pastYVals[act_window_idx][index] = yVals[(row['Stock'], lastDay)]
			pastYDist[act_window_idx][index] = actDay_orig - lastDay
			semi_pastYDist[act_window_idx][index] = actDay - lastDay
			if (act_window_idx < max_window_idx):
				return update_past(lastDay - 1, actDay_orig, act_window_idx + 1, max_window_idx, minDay, filledDays, yVals, pastYVals, pastYDist, index, row, semi_pastYDist)
			break
		lastDay = lastDay - 1
	return (pastYVals, pastYDist, semi_pastYDist)

def augment_dataframe(window, dataframe, minDay, maxDay, filledDays, yVals, orig_features):
	num_entries = 0
	for index, row in dataframe.iterrows():
		num_entries = num_entries + 1

	pastYVals = []
	futureYVals = []
	pastYDist = []
	semi_pastYDist = []
	futureYDist = []
	semi_futureYDist = []
	weightedAvg = []
	semi_weightedAvg = []
	unweightedAvg = []
	for i in range(window):
		pastYVals.append(np.zeros(num_entries))
		futureYVals.append(np.zeros(num_entries))
		pastYDist.append(np.zeros(num_entries))
		futureYDist.append(np.zeros(num_entries))
		semi_pastYDist.append(np.zeros(num_entries))
		semi_futureYDist.append(np.zeros(num_entries))
		weightedAvg.append(np.zeros(num_entries))
		semi_weightedAvg.append(np.zeros(num_entries))
		unweightedAvg.append(np.zeros(num_entries))

	for index, row in dataframe.iterrows():
		print(index)
		actDay = int(row['Day'])

		for i in range(window):
			futureYDist[i][index] = 999
			pastYDist[i][index] = 999
			semi_futureYDist[i][index] = 999
			semi_pastYDist[i][index] = 999

		(futureYVals, futureYDist, semi_futureYDist) = update_future(actDay, actDay, 0, window - 1, maxDay, filledDays, yVals, futureYVals, futureYDist, index, row, semi_futureYDist)
		(pastYVals, pastYDist, semi_pastYDist) = update_past(actDay, actDay, 0, window - 1, minDay, filledDays, yVals, pastYVals, pastYDist, index, row, semi_pastYDist)

		sumValPast = 0
		sumValNext = 0
		sumWeightPast = 0
		sumWeightNext = 0
		semi_sumValPast = 0
		semi_sumValNext = 0
		semi_sumWeightPast = 0
		semi_sumWeightNext = 0
		sumValPast_unweighted = 0
		sumValNext_unweighted = 0
		sumWeightPast_unweighted = 0
		sumWeightNext_unweighted = 0
		for i in range(window):
			sumValPast = sumValPast + (1/float(pastYDist[i][index])) * pastYVals[i][index]
			sumValNext = sumValNext + (1/float(futureYDist[i][index])) * futureYVals[i][index]
			sumWeightPast = sumWeightPast + (1/ float(pastYDist[i][index]))
			sumWeightNext = sumWeightNext + (1 / float(futureYDist[i][index]))

			semi_sumValPast = semi_sumValPast + (1/float(semi_pastYDist[i][index])) * pastYVals[i][index]
			semi_sumValNext = semi_sumValNext + (1/float(semi_futureYDist[i][index])) * futureYVals[i][index]
			semi_sumWeightPast = semi_sumWeightPast + (1/ float(semi_pastYDist[i][index]))
			semi_sumWeightNext = semi_sumWeightNext + (1 / float(semi_futureYDist[i][index]))
			if (sumWeightPast + sumWeightNext) > 0:
				weightedAvg[i][index] = (sumValPast + sumValNext) / (sumWeightPast + sumWeightNext)
				semi_weightedAvg[i][index] = (semi_sumValPast + semi_sumValNext) / (semi_sumWeightPast + semi_sumWeightNext)

			if pastYDist[i][index] != 999:
				sumValPast_unweighted = sumValPast_unweighted + pastYVals[i][index]
				sumWeightPast_unweighted = sumWeightPast_unweighted + 1
			if futureYDist[i][index] != 999:
				sumValNext_unweighted = sumValNext_unweighted + futureYVals[i][index]
				sumWeightNext_unweighted = sumWeightPast_unweighted + 1
			if (sumWeightPast_unweighted + sumWeightNext_unweighted) > 0:
				unweightedAvg[i][index] = (sumValPast_unweighted + sumValNext_unweighted) / (sumWeightPast_unweighted + sumWeightNext_unweighted)

	features = list(orig_features)
	for i in range(window):
		dataframe['pastY' + str(i+1)]=pastYVals[i]
		dataframe['pastYDist' + str(i+1)]=pastYDist[i]
		dataframe['futureY' + str(i+1)]=futureYVals[i]
		dataframe['futureYDist' + str(i+1)]=futureYDist[i]
		dataframe['weightedAvg' + str(i+1)] = weightedAvg[i]
		dataframe['semi_weightedAvg' + str(i+1)] = semi_weightedAvg[i]
		dataframe['unweightedAvg' + str(i+1)] = unweightedAvg[i]

		features.append('pastY' + str(i+1))
		features.append('pastYDist' + str(i+1))
		features.append('futureY' + str(i+1))
		features.append('futureYDist' + str(i+1))
		features.append('weightedAvg' + str(i+1))
		features.append('semi_weightedAvg' + str(i+1))
		features.append('unweightedAvg' + str(i+1))
	return (dataframe, features)


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

window = 10

orig_features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']
print("Part 2...")
(df, aug_features) = augment_dataframe(window, df, minDay, maxDay, filledDays, yVals, orig_features)
df.to_csv("train_augmented.csv")
print("Part 3...")

aug_features_plus_y = ['y']
for f in aug_features:
	aug_features_plus_y.append(f)

aug_features_plus_y = [ x for x in aug_features_plus_y if x not in orig_features]
for i in range(window):
	aug_features_plus_y.remove('pastY' + str(i+1))
	aug_features_plus_y.remove('pastYDist' + str(i+1))
	aug_features_plus_y.remove('futureY' + str(i+1))
	aug_features_plus_y.remove('futureYDist' + str(i+1))

g = sns.heatmap(df[aug_features_plus_y].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
print(df[aug_features_plus_y].corr())
plt.show()
print("Part 4...")

for index, row in df_test.iterrows():
	if row['Stock'] not in filledDays:
		filledDays[row['Stock']] = set()
		print("Stock number " + str(row['Stock']) + " does not appear in the training data.")

(df_test, aug_features_test) = augment_dataframe(window, df_test, minDay, maxDay, filledDays, yVals, orig_features)
df_test.to_csv("test_augmented.csv")
