#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:14:26 2022

@author: garimasharma
"""
# import libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split

#%% load data set and get X and Y data

breast_cancer_data = pd.read_csv('breast_cancer.csv', header=None)

Y = breast_cancer_data[30].to_numpy()
X = breast_cancer_data.drop(columns=30).to_numpy()

#%% split the X and Y into train and test set


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

#%% 

dt_model = DecisionTreeClassifier()

dt_model = dt_model.fit(X_train, Y_train)

Y_pred = dt_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(Y_test, Y_pred)
print(f"The accuracy of the Decision tree is {accuracy}")

class_report = classification_report(Y_test, Y_pred)
print(f"The classification report is: {class_report}")

#%% visulaize the tree

from sklearn import tree
tree.plot_tree(dt_model)




