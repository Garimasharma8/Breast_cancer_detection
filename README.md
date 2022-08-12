# Breast_cancer_detection

Objective: Our main objective is to classify data labelled as breast cancer positive (class 1) from the data labelled as breast cancer negative (class 0). 
The dataset has 30 features and 569 rows of data which is labelled as class 1 or class 0. 

# Methodology:
we have tried three different classifiers: 
a) Logistic Regression
b) Decision Trees
c) Random Forest

# Approach A : Classifing using Logistic regression: 
we need to convert the linear regression line into the sigmoid curve. That means we need to map the line y = mx + c to a sigmoid line. The steps to do this are:

1. The eq. for linear regression is y = mx + c, has boundaries -infinity to +infinity. The logistic regression has to be between 0 and 1. 
2. Divide y by (1-y), now the range is from 0 to +infinity. Because when y = 0 in Y/(1-Y), the output is 0, and when y = 1 the output is infinity. 
3. Now for logistic regression, we got the range [0, infinity], but still, this is not enough, we need to pull down the upper range from infinity to 1. To do that we apply log to the equation, the updated equation is log(y / 1-y). The range is now [0,1].

Steps to model Logistic regression are:
1. Import libraries: pandas, numpy, sklearn.
2. Check for missing rows, and handle it accordingly. 
3. Split the data into train and test set.
4. Fit the train data to the classifier. The logistic regression classifier is imported from sklean.linear_model. 
5. Check the predicted labels for test data, and generate the classification report.

The output of Logistic regression model is: 
![image](https://user-images.githubusercontent.com/97305078/184277602-3f244b04-46b6-48d1-889d-f423b6ea9225.png)

# Approach B: Classifing using DT
I have implemented DT in python by using sklearn module. The DT uses gini index by default to build decision tree- choosing root node and relevant attributes for tree splitting. 

The implementation is pretty straight forward. The general steps to perform any classification task are: 

1. step 1: Import python libraries: pandas, sklearn, and numpy.
2. Step 2 and Step 3: Import Dataset, get X (instances and features) and Y (labels).  Remove rows with missing data or handle/fill it by using some predefined rules (e.g. mean value or mode value). In this dataset, we have no missing values, hence we haven't removed any row from the data frame. The last column of the data have labels (i.e. Y) and could be extracted from the data itself. The breast cancer data has 569 instances and 30 features which creates the variable X. So, size of X is (569,30) and size of Y is (569,).
3. Step 4: Split X and Y into X_train, X_test, Y_train, and Y_test using train_test_split method (imported from sklearn.model_selection). Now the size of X_train is (455,30) and X_test is (114,30). Similarly, the size of Y_train is (455,) and Y_test is (114,). 
4. Step 5: Define the classifier, and fit Training data on it. The classifier is DecisionTreeClassifier (imported from sklearn.tree).
5. Step 6: Pass X_test into the trained classifier and find Y_predict. 
6. Step 7: Compare Y_test and Y_predict using accuracy rate or classification report.

The accuracy of the Decision tree is 0.956140350877193
The classification report is:     

![image](https://user-images.githubusercontent.com/97305078/184276504-1e0ea32a-b798-441c-867d-594e9d6a7d73.png)


# Approach B: Classifing using Random forest:
We have implemented logistic regression on the same dataset and get an accuracy rate of 90%. Now let’s see if random forests will provide us better accuracy or not. 

Steps to model the random forest are shown below: 
1. Import libraries: pandas, numpy, and sklearn.
2. Getting features (X) and target (Y) from the dataset, in the dataset the last column is target columns, where 0: no breast cancer and 1: breast cancer diagnosed. 
3. Use train_test_split to split the dataset into training and testing se. We have used 30% of data as test set. 
4. Create a random forest with 5 DT and these tree trained themselves as per gini index. We can also use entropy as the criterion. Train the model on the scaled training set. 
5. Once the model is trained, we can predict the outcomes on scaled test data. 
6. Generate classification report

The output of the random forest classifier is:

![image](https://user-images.githubusercontent.com/97305078/184278451-b39824de-c96f-4832-99c3-d650ccd9ecce.png)

