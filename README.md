# Breast_cancer_detection

I have implemented DT in python by using sklearn module. The DT uses gini index by default to build decision tree- choosing root node and relevant attributes for tree splitting. 

The implementation is pretty straight forward. The general steps to perform any classification task are: 

1. Import libraries
2. Import Dataset, 
3. Remove rows with missing data or handle/fill it by using some predefined rules (e.g. mean value or mode value). Get X (instances and features) and Y (labels). 
4. Split X and Y into X_train, X_test, Y_train, and Y_test.
5. Define the classifier, and fit Training data on it. 
6. Pass X_test into the trained classifier and find Y_predict. 
7. Compare Y_test and Y_predict using accuracy rate or classification report.

The details of the steps are: 

1. step 1: Import python libraries: pandas, sklearn, and numpy.
2. Step 2 and Step 3: Import Dataset, get X (instances and features) and Y (labels).  Remove rows with missing data or handle/fill it by using some predefined rules (e.g. mean value or mode value). In this dataset, we have no missing values, hence we haven't removed any row from the data frame. The last column of the data have labels (i.e. Y) and could be extracted from the data itself. The breast cancer data has 569 instances and 30 features which creates the variable X. So, size of X is (569,30) and size of Y is (569,).
3. Step 4: Split X and Y into X_train, X_test, Y_train, and Y_test using train_test_split method (imported from sklearn.model_selection). Now the size of X_train is (455,30) and X_test is (114,30). Similarly, the size of Y_train is (455,) and Y_test is (114,). 
4. Step 5: Define the classifier, and fit Training data on it. The classifier is DecisionTreeClassifier (imported from sklearn.tree).
5. Step 6: Pass X_test into the trained classifier and find Y_predict. 
6. Step 7: Compare Y_test and Y_predict using accuracy rate or classification report.

The accuracy of the Decision tree is 0.956140350877193
The classification report is:     

![image](https://user-images.githubusercontent.com/97305078/184276504-1e0ea32a-b798-441c-867d-594e9d6a7d73.png)





