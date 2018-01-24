#! /usr/env/python

from blackbox import *

features = ['Market', 'Stock', 'x0', 'x1', 'x2', 'x3A', 'x3B', 'x3C', 'x3D', 'x3E', 'x4', 'x5', 'x6']

df = pd.read_csv('train.csv', index_col=0)
df.head()
df_test = pd.read_csv('test.csv', index_col=0)
df_test = df_test.fillna(0)
df_test.head()

weights = np.concatenate(df[['Weight']].values, axis = 0)
classifier = Classifier()
classifier.set_features(features)
classifier.test_size = 0
classifier.set_csv_file('train.csv')

outfile = open("bayesian_ridge_all.csv", "w")
outfile.write("Index,y\n")

X_test = df_test[features].values
y_predicted = classifier.best_model.predict(X_test)

i = 0

for index, row in df_test.iterrows():
	prediction = y_predicted[i]
	outfile.write(str(index) + "," + str(prediction) + "\n")
	i = i + 1
outfile.close()
