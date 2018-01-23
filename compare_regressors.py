#! /usr/env/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.gaussian_process import GaussianProcessRegressor # no sample weights
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ARDRegression # no sample weights
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import TheilSenRegressor # no sample weights
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor # no sample weights
from sklearn.neural_network import MLPRegressor # no sample weights

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def plot_distribution(data, title, column_label, do_log):
	dataframe = data.copy()
	if do_log:
		# Apply log to reduce skewness distribution
		dataframe[column_label] = dataframe[column_label].map(lambda i: np.log(i) if i > 0 else 0)
	# Explore distribution in dataset
	g = sns.distplot(dataframe[column_label], color="m", label="Skewness : %.2f"%(dataframe[column_label].skew()), ax=ax)
	g = g.legend(loc="best")
	ax.set_title(title)
	plt.savefig(title + '.png')
	plt.close()


features = ['Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']

df = pd.read_csv('train.csv', index_col=0)
df = df.fillna(0) # replace NaN entries
#weights = []
#for index, row in df.iterrows():
#	weights.append(float(row['Weight']))
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0) # replace NaN entries

X = df[features]
Y = df['y']

# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
#g = sns.heatmap(df[['y', 'Market', 'Day', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


ax = plt.axes()

# Explore x0 distribution in train dataset ----> x0 seems extremely skewed, there are some extreme outliers!
#g = sns.distplot(df["x0"], color="m", label="Skewness : %.2f"%(df["x0"].skew()), ax=ax)
#g = g.legend(loc="best")
#ax.set_title("x0 distribution in train dataset")

# Explore x0 distribution in test dataset ----> x0 seems extremely skewed, there are some extreme outliers!
#g = sns.distplot(df_test["x0"], color="m", label="Skewness : %.2f"%(df_test["x0"].skew()), ax=ax)
#g = g.legend(loc="best")
#ax.set_title("x0 distribution in test dataset")

for f in features:
	plot_distribution(df, f + " distribution in train dataset", f, False)
	plot_distribution(df, "log " + f + " distribution in train dataset", f, True)
	plot_distribution(df_test, f + " distribution in test dataset", f, False)
	plot_distribution(df_test, "log " + f + " distribution in test dataset", f, True)

