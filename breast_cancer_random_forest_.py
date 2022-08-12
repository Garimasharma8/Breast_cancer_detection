#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:40:59 2022

@author: garimasharma
random forest on breast_cancer dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

breast_cancer_data = pd.read_csv('breast_cancer.csv', header=None)

print(breast_cancer_data.head())

#%% Get X and Y from the dataset 

X = breast_cancer_data.iloc[:, range(0,30)] # select all columns from 0 to 29
Y = breast_cancer_data.iloc[:, 30]

print(breast_cancer_data.info)

#%% Splitting the dataset into traning and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print(f"The size of training set is {X_train.shape}")
print(f"The size of test set is {X_test.shape}")

#%% Feature scaling using standardscaler 

from sklearn.preprocessing import StandardScaler

SC = StandardScaler()

X_training_scaled = SC.fit_transform(X_train)
X_test_scaled = SC.fit_transform(X_test)

#%% Train Random forest on X_training_scaled dataset

from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators=5, criterion='gini', random_state=0)

RF_model.fit(X_training_scaled,Y_train)

#%% perfrom prediction using test data

Y_pred = RF_model.predict(X_test_scaled)

#%% classification response 

from sklearn.metrics import classification_report, accuracy_score

RF_performance = classification_report(Y_test, Y_pred)
accuracy_RF = accuracy_score(Y_test, Y_pred)
print(f"The accuarcy rate of Random forest is {accuracy_RF}")

# we got 95% of accuracy - better than logistic regression

#%% plot trees- I am trying but not done yet

import PIL
import pydotplus
from glob import glob
from IPython.display import display, Image
from sklearn.tree import export_graphviz


def save_trees_as_png(clf, iteration, feature_name, target_name):
    file_name = "Breast_cancer" + str(iteration) + ".png"
    dot_data = export_graphviz(clf, out_file=None, feature_names=feature_name,
                               class_names=target_name,
                               rounded=(True),
                               proportion=(False),
                               precision=2,
                               filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(file_name)
    print("Decision Tree {} saved as png file".format(iteration+1))
    






