# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:46:42 2023

@author: casper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Reading Data
veriler = pd.read_csv('musteriler.csv')
X = veriler.iloc[:,3:].values

#K-Means
from sklearn.cluster import KMeans
kmeans = KMeans( n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

#Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

Y_pred = ac.fit_predict(X)

plt.scatter(X[Y_pred==0,0],X[Y_pred==0,1],c='red')
plt.scatter(X[Y_pred==1,0],X[Y_pred==1,1],c='blue')
plt.scatter(X[Y_pred==2,0],X[Y_pred==2,1],c='green')
plt.scatter(X[Y_pred==3,0],X[Y_pred==3,1],c='black')
plt.scatter(X[Y_pred==4,0],X[Y_pred==4,1],c='yellow')
plt.show()

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))



























