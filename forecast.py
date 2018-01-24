#! /usr/env/python

from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model

from sklearn.naive_bayes import GaussianNB

def wavg(group, avg_name, name2, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name][name2]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return 0

df = pd.read_csv('train.csv', index_col=0)
df.head()

df_test = pd.read_csv('test.csv', index_col=0)
df_test.head()


y_sum_per_stock = {}
weights_per_stock = {}
for index, row in df.iterrows():
	if not row['Stock'] in y_sum_per_stock:
		y_sum_per_stock[row['Stock']] = 0.0
		weights_per_stock[row['Stock']] = 0.0
	y_sum_per_stock[row['Stock']] = y_sum_per_stock[row['Stock']] + (float(row['y']) * float(row['Weight']))
	weights_per_stock[row['Stock']] = weights_per_stock[row['Stock']] + float(row['Weight'])

for index, row in df.iterrows():
	y_sum_per_stock[row['Stock']] = y_sum_per_stock[row['Stock']] / weights_per_stock[row['Stock']]
#now y_sum_per_stock contains the weighted averages

outfile = open("weighted_avg.csv", "w")
outfile.write("Index,y\n")
for index, row in df_test.iterrows():
	if row['Stock'] in y_sum_per_stock:
		outfile.write(str(index) + "," + str(y_sum_per_stock[row['Stock']]) + "\n")
	else:
		outfile.write(str(index) + "," + "0" + "\n")
outfile.close()

#model = sklearn.linear_model.LinearRegression()
#model = sklearn.neighbors.KNeighborsRegressor()
#X = df[['x4']]
#y = df['y']

#weights = np.concatenate(df[['Weight']].values, axis = 0)
#print(weights)

#stock_mean = df.groupby('Stock')['y'].mean()

#print(df.groupby('Stock')['y'])

#weighted_stock_mean = df.groupby('Stock')['y'].apply(wavg, "Stock", "y", "Weight")

#weighted_stock_mean.head()

#yp = df_test['Stock'].map(weighted_stock_mean).rename('y')
#yp.head()

#yp.fillna(0).to_csv('weighted_mean_model.csv', header=True)


#model.fit(X, y, sample_weight = weights)

#print('alpha:  \t', model.intercept_)
#print('beta:   \t', model.coef_[0])

#print('r^2 score:\t', model.score(X, y))

#sns.regplot(X[::100], y[::100]);

#yp = pd.Series(model.predict(df_test[['x4']])).rename('y')
#yp.index.name = 'Index'
#yp.head()

#yp.to_csv('mymodel2.csv', header=True)
