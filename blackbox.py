#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from collections import Counter

import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from imblearn.ensemble import EasyEnsemble

from sklearn.externals import joblib

from sklearn import linear_model
from sklearn import svm

import os


class Classifier:
  def __init__(self):
    self.features = []

    self.test_size = 0.33
    
    #self.models = [linear_model.LinearRegression(), linear_model.SGDRegressor(), linear_model.BayesianRidge(), linear_model.LassoLars(), linear_model.PassiveAggressiveRegressor(), linear_model.TheilSenRegressor(), svm.SVR(), linear_model.ARDRegression()]
    #self.modelnames = ['LinearRegression', 'SGDRegressor', 'BayesianRidge', 'LassoLars', 'PassiveAggressiveRegressor', 'TheilSenRegressor', 'SVR', 'ARDRegression']
    
    self.models = [linear_model.LinearRegression()]
    self.modelnames = ['LinearRegression']

    self.best_model = LinearRegression()

  def read_data(self, input_file):
    print("Reading data...")
    df = pd.read_csv(input_file, index_col=0)
    df = df.fillna(0)
    #X = df[self.features].values
    Y = df['y'].values

    features_with_weights = list(self.features)
    features_with_weights.append('Weight')
    X_with_weights = df[features_with_weights].values

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_with_weights, Y, test_size=self.test_size)

    # remove the weights in X_train
    X_train = np.delete(X_train, np.s_[len(self.features):len(self.features) + 1], axis=1)

    weights = []
    # remove the weights in X_test
    for sublist in X_test:
	weights.append(sublist[len(sublist) - 1])
    	sublist = sublist[:-1]

    # remove the weights in X_test
    X_test = np.delete(X_test, np.s_[len(self.features):len(self.features) + 1], axis=1)

    print(X_train[0])

    return (X_train, X_test, y_train, y_test, weights)
    
  #takes std::vector<std::string>
  def set_features(self, features):
    self.features = features

  def choose_best_classifier(self, X_train, X_test, y_train, y_test, bestError, weights):
    expected = y_test
    best_error = bestError
    best_num = 0
    
    for i in range(len(self.models)):
        # Fit the model
        print("Fitting model: " + self.modelnames[i] + "...")
        self.models[i].fit(X_train, y_train)

	if (self.test_size > 0):
		# Evaluate the model
		print("Predicting regression results with model: " + self.modelnames[i] + "...")
		predicted = self.models[i].predict(X_test)
		# print some metrics
		error = sklearn.metrics.mean_squared_error(expected, predicted, sample_weight = weights)	
		print("The mean squared error for model " + self.modelnames[i] + " with sample weights is: " + str(error))
		if error < best_error:
		    self.best_model = self.models[i]
		    best_error = error
		    best_num = i
	else:
		self.best_model = self.models[i]
		best_error = 0
		best_num = i
    print("And the winner is: " + self.modelnames[best_num])
    return best_error
      
  #takes std::string
  def set_csv_file(self, csv_path):
    print(csv_path)
    (X_train, X_test, y_train, y_test, weights) = self.read_data(csv_path)

    bestF = 999999
    # Do not prebalance the dataset before training, use sample weights
    bestF = self.choose_best_classifier(X_train, X_test, y_train, y_test, bestF, weights)
    
  #takes std::string
  def store_classifier(self, filename):
      joblib.dump(self.best_model, filename, compress=9)
      print("Stored classifier.")
      
  #takes std::string
  def load_classifier(self, filename):
      self.best_model = joblib.load(filename)
      print("Loaded classifier.")

  def predict(self, feature_vector):
    matrix = []
    matrix.append(feature_vector)
    return self.best_model.predict(matrix)[0]

