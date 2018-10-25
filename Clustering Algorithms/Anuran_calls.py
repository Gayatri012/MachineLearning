# -*- coding: utf-8 -*-
import pandas as pd
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Import data
data=pd.read_csv(sys.argv[1])
data=data.iloc[:,1:22]

# calculates the centroid of data
center=data.mean(0)

# Total sum of the squares (TSS) for the data
sum1=0
for i in range(0,len(data)-1):
    for j in range(0,21):

        sum1=sum1+(data.iat[i,j]-center[j])**2
    
kmeans_total = KMeans(n_clusters=1)
kmeans_total.fit(data)
y_kmeans_total = kmeans_total.predict(data)
tss = kmeans_total.inertia_

print("K means clustering for anuran calls")
print("# of clusters","   Ratio of twss/tss")
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    twss = kmeans.inertia_
   
    ratio = twss/tss
    print(i, "\t\t", ratio)


print("Hierarchical clustering for anuran calls")
print("# of clusters","   Ratio of twss/tss")

for i in range(1,11):
    model = AgglomerativeClustering(n_clusters=i) 
    model.fit(data)
    twss_hCluster = [0] * i
    label_hCluster = model.labels_
    
    data['label']=label_hCluster
    centers = np.array([data[label_hCluster == j].mean(0) for j in range(i)])

     #removing the label column
    centers=centers[:,:-1]
    for k in range(0,i):
       for p in range(0,len(data[data['label'] == k])-1):
           squares = (data[data['label'] == k].iloc[p,:-1] - centers[k]) ** 2
           twss_hCluster[k]+= np.sum(squares)
    twss_hCluster_tot = np.sum(twss_hCluster)
    
    print(i, "\t\t", twss_hCluster_tot/sum1)
    
    data=data.drop(['label'], axis=1)
   
    
    
'''
# H clustering using scipy    
Z = linkage(data, 'ward')
c, coph_dists = cophenet(Z, pdist(data))
print(c)
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
'''

print("Gaussian Mixture Model for anuran calls")
print("# of clusters","   Ratio of twss/tss")

for i in range(1,11):
    twss1 = [0] * i
    gmm = GaussianMixture(n_components=i).fit(data)
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
   