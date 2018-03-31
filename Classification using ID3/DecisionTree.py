# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:07:55 2018


"""

# Import the library
from sklearn import tree
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import graphviz

# We load the dataset ...
dataset = pd.read_csv(sys.argv[1])
x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[0:-1,0:6], dataset.iloc[0:-1,6:7], test_size=0.25, random_state=0)

# Here we build the Decision Tree classifier
clf = tree.DecisionTreeClassifier()

data = pd.get_dummies(x_train, columns=['V1','V2','V3','V4','V5','V6'])
target= pd.get_dummies(y_train, columns=['V7']) 
data_test = pd.get_dummies(x_test, columns=['V1','V2','V3','V4','V5','V6'])
target_test = pd.get_dummies(y_test, columns=['V7'])
clf = clf.fit(data,target)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=data.columns.values,  
                         class_names=target.columns.values,  
                         filled=True, rounded=True,  
                         special_characters=True)

graph = graphviz.Source(dot_data)

graph.render("DecisionTreeGraph.gv", view=True)

graph

# Here we predict a new value
prediction=clf.predict(data_test)

pred = pd.DataFrame(data = prediction, columns = target.columns.values)
output_pred = pred.idxmax(axis=1)

pred_for_cm = []

print("Predictions: ")
for i in range(len(output_pred)):
    pred_for_cm.append(output_pred[i][3:])
    print(output_pred[i][3:])

print("Accuracy Score of Prediction: " , accuracy_score(target_test, prediction))

print("Confusion matrix for Prediction: ")
print( confusion_matrix(y_test, pred_for_cm))
