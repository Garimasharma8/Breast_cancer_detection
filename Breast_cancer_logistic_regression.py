#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:08:59 2022

@author: garimasharma
This is the script file for logostic regression on breast cancer file. We need to classify if a person 
is having breast cancer or not ( 0: no cancer, 1: cancer)
"""
import pandas as pd      # for data handling
from sklearn.linear_model import LogisticRegression     # for model import  
from sklearn.metrics import classification_report, accuracy_score  # for metrics

#%% Import the data and explore it 

data  = pd.read_csv('breast_cancer.csv', header=None)
print(f"The size of the data is {data.shape}, it has {data.shape[0]} rows and {data.shape[1]} columns")
print(data.head())

#%% check for missing data

missing_data = data.isnull().sum()

#%% train data and test data , we can also use train_test_split

X = data.iloc[:, range(0,30)] # select all columns from 0 to 29
Y = data.iloc[:, 30]

X_train = X[:400]
X_test = X[400:569]

Y_train = Y[:400]
Y_test = Y[400:569]

#%% design the model, fit on training data, and test on testing data



logistic_model = LogisticRegression()

logistic_model.fit(X_train, Y_train)

logistic_predict = logistic_model.predict(X_test)



class_report = classification_report(Y_test, logistic_predict)
accuracy = accuracy_score(Y_test,logistic_predict)

print(f"The classification report is: {class_report}")
print(f"The accuracy of the logistic regreesion model is {accuracy}")

