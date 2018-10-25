# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:17:36 2018

@author: Sn
"""

import pandas as pd
import sys
#import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture


data=pd.read_csv(sys.argv[1],sep=';')
data = data.dropna()
type_data=pd.get_dummies(data.iloc[:,1])

# Adding dummy variable values to the data
data['Link']=type_data.iloc[:,0]
data['Photo']=type_data.iloc[:,1]
data['Status']=type_data.iloc[:,2]
data['Video']=type_data.iloc[:,3]

# Drop category type column after adding its corresponding dummy variables
data=data.drop(['Type'], axis=1)

# calculates the centroid of data
center=data.mean(0)

# Total sum of the squares (TSS) for the data
sum1=0
for i in range(0,len(data)-1):
    for j in range(0,21):

        sum1=sum1+(data.iat[i,j]-center[j])**2

# To find TSS for K means using sklearn's in built attribute for the entire data
kmeans_total = KMeans(n_clusters=1)
kmeans_total.fit(data)
y_kmeans_total = kmeans_total.predict(data)
tss = kmeans_total.inertia_

print("K means clustering for Facebook Posts")
print("# of clusters","   Ratio of twss/tss")
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    twss = kmeans.inertia_

    ratio = twss/tss

    print(i, "\t\t", ratio)
    
print("\nHierarchical clustering for Facebook Posts")
print("# of clusters","   Ratio of twss/tss")

for i in range(1,11):
    model = AgglomerativeClustering(n_clusters=i) 
    model.fit(data)
    
    twss_hCluster = [0] * i
    label_hCluster = model.labels_
    data['label']=label_hCluster
    
    # Centroid for each cluster
    centers = np.array([data[label_hCluster == j].mean(0) for j in range(i)])
     #removing the label column before finding TWSS as it is not a part of data
    centers=centers[:,:-1]
    
    #Calculate Total within sum of squares (TWSS)
    for k in range(0,i):
       for p in range(0,len(data[data['label'] == k])-1):
           squares = (data[data['label'] == k].iloc[p,:-1] - centers[k]) ** 2
           twss_hCluster[k]+= np.sum(squares)
    twss_hCluster_tot = np.sum(twss_hCluster)
    
    print(i, "\t\t", twss_hCluster_tot/sum1)
    
    data=data.drop(['label'], axis=1)
    

print("\nGaussian Mixture Model for Facebook Posts")
print("# of clusters","   Ratio of twss/tss")

for i in range(1,11):
    twss1 = [0] * i
    # reg_covar is increased from its default value to give a non negative covariance
    gmm = GaussianMixture(n_components=i,reg_covar=0.0001).fit(data)
    labels = gmm.predict(data)
    data['labels']=labels
    centers = np.array([data[labels == j].mean(0) for j in range(i)])
     #removing the label column
    centers=centers[:,:-1]
    for k in range(0,i):
       for p in range(0,len(data[data['labels'] == k])-1):
           squares = (data[data['labels'] == k].iloc[p,:-1] - centers[k]) ** 2
           twss1[k]+= np.sum(squares)
    twss1_tot = np.sum(twss1)
    
    print(i, "\t\t", twss1_tot/sum1)
    
    data=data.drop(['labels'], axis=1)
   
    