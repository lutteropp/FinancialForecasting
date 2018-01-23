#! /usr/env/python

import pandas as pd
import matplotlib.pyplot as plt
import math

features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']

df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries

minY = 9999999
maxY = -9999999
sumY = 0
numY = 0
for index, row in df.iterrows():
	sumY = sumY + row['y']
	numY = numY + 1
	if row['y'] < minY:
		minY = row['y']
	if row['y'] > maxY:
		maxY = row['y']

meanY = sumY / float(numY)

variance = 0
for index, row in df.iterrows():
	variance = variance + (meanY - row['y']) * (meanY - row['y'])

print("minY: ", minY)
print("maxY: ", maxY)
print("meanY: ", meanY)
print("variance: ", variance)
print("numY: ", numY)

ySorted = df['y']
ySorted = ySorted.sort()

print("median: ", ySorted[len(ySorted) / 2])

nbins = (maxY - minY) * (numY ** (1/float(3))) / (3.49 * math.sqrt(variance))

print("nbins: ", int(round(nbins)))

plt.hist(df['y'], int(round(nbins)))
plt.show()
