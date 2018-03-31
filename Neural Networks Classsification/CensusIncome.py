# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:04:49 2018


"""

# Import Statements
import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import cnm_plot
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

# Read the header info file
headerFromFile = pd.read_csv(sys.argv[1], sep=':', header=None)

# Separating header from description
headers = headerFromFile.iloc[:,0:1]

# Adding a value for the Prediction Column
lastRow = pd.DataFrame(["Income"])

# If header file does not contain the Prediction Column Name, add it
if (len(headers) is 14):
    headers = pd.concat([headers, lastRow])

# Load train data from file
data = pd.read_csv(sys.argv[2], header=None)
# Load test data from file
test = pd.read_csv(sys.argv[3], header=None)

# Assign headers to the columns
data.columns = headers
test.columns = headers

# Data cleaning
data = data[data.iloc[:, 1] != " ?"]
data = data[data.iloc[:, 6] != " ?"]

test = test[test.iloc[:, 1] != " ?"]
test = test[test.iloc[:, 6] != " ?"]

# Using only important attributes
data.drop(data.columns[[2, 4, 10, 11, 13]], axis=1, inplace=True)
test.drop(test.columns[[2, 4, 10, 11, 13]], axis=1, inplace=True)

# Minimizing the features - Combining some features to same class
data.replace(to_replace = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], 
             value = ['Bachelors', 'Some-college', 'School', 'Some-college', 'Bachelors', 'Assoc', 'Assoc', 'School', 'School', 'School', 'Bachelors', 'School', 'School', 'Doctorate', 'School', 'School'], 
             inplace = True, regex = True)

test.replace(to_replace = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], 
             value = ['Bachelors', 'Some-college', 'School', 'Some-college', 'Bachelors', 'Assoc', 'Assoc', 'School', 'School', 'School', 'Bachelors', 'School', 'School', 'Doctorate', 'School', 'School'], 
             inplace = True, regex = True)

data.replace(to_replace = ['Never-married', 'Widowed', 'Married-spouse-absent', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
             value = ['Separated', 'Divorced', 'Separated',  'Family', 'Family', 'Family', 'Not-in-family', 'Family', 'Not-in-family'], 
             inplace = True, regex = True)


test.replace(to_replace = ['Never-married', 'Widowed', 'Married-spouse-absent', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
             value = ['Separated', 'Divorced', 'Separated',  'Family', 'Family', 'Family', 'Not-in-family', 'Family', 'Not-in-family'], 
             inplace = True, regex = True)



# Creating dummy variable for categorical data
train_data = pd.get_dummies(data.iloc[:,:-1])
train_target = pd.get_dummies(data.iloc[:,-1])

test_data = pd.get_dummies(test.iloc[:,:-1])
test_target = pd.get_dummies(test.iloc[:,-1])


# Combining multiple columns categorical value of train target to a single column
for i, col in enumerate(train_target.columns.tolist(), 1):
    train_target.loc[:, col] *= i
    
train_target_svm = train_target.sum(axis=1)

# Combining multiple columns categorical value of test target to a single column
for i, col in enumerate(test_target.columns.tolist(), 1):
    test_target.loc[:, col] *= i
    #print(i)

test_target_cumulative = test_target.sum(axis=1)

# Scaling
sm = SMOTE(random_state=12, ratio = "minority")
x_train_res, y_train_res = sm.fit_sample(train_data, train_target_svm)

# If any categorical value is present in the train data but absent in test data, add the category with value 0
data_dummy_columns = list(train_data.columns.values)
test_dummy_columns = list(test_data.columns.values)

for i in data_dummy_columns:
    if i not in test_dummy_columns:
        test_data.insert(data_dummy_columns.index(i), i, 0)


# Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(5,10,5), max_iter=200)

# Fit the model ... learn the weights
mlp.fit(x_train_res, y_train_res)

# Predict using the model
predictions = mlp.predict(test_data)

# Confusion matrix
predictions = pd.DataFrame(predictions)
print("Neural Network")
print("---------------")

cf = confusion_matrix(test_target_cumulative, predictions)

cnm_plot.plot_confusion_matrix(cf,classes=list(test_target.columns.values), title = "Neural Network")

print("Accuracy Score:", mlp.score(test_data, test_target_cumulative))
print("Confusion matrix")
print(cf)

#print("Accuracy score: ", accuracy_score(test_target_cumulative, predictions) )


#SVM - using radial kernel
print("\nSVM")
print("------")
svc_radial = svm.SVC(C = 100)
svc_radial.fit(train_data, train_target_svm)
predicted= svc_radial.predict(test_data)

cnf_matrix = confusion_matrix(test_target_cumulative, predicted)
cnm_plot.plot_confusion_matrix(cnf_matrix,classes=list(test_target.columns.values), title = "SVM")

print("Accuracy score: ", accuracy_score(test_target_cumulative, predicted) )
print("Confusion matrix")
print(cnf_matrix)


#KNN
print("\nKNN")
print("-----")
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(train_data, train_target_svm)

predictions_KNN = KNN.predict(test_data)
print("Accuracy Score", KNN.score(test_data, test_target_cumulative))

confusion_matrix_KNN = confusion_matrix(test_target_cumulative, predictions_KNN)
cnm_plot.plot_confusion_matrix(confusion_matrix_KNN,classes=list(test_target.columns.values), title = "KNN")
print("Confusion matrix")
print(confusion_matrix_KNN)


cnm_plot.closeFile()

