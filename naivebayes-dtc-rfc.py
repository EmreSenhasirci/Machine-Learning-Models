# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:53:00 2023

@author: casper
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
veriler = pd.read_csv('veriler.csv')
print(veriler)
boy = veriler.iloc[:,1:2].values
kilo = veriler.iloc[:,2:3].values
x= veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('-----------------')
print('GNB')
print(cm)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Criterion = Entropy
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('-----------------')
print('DTC, ENTROPY')
print(cm)
# Criterion = gini
dtc = DecisionTreeClassifier(criterion = 'gini')
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('-----------------')
print('DTC, GINI')
print(cm)

from sklearn.ensemble import RandomForestClassifier
#Random Forest - Entropy
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('-----------------')
print('RFC, ENTROPY')
print(cm)
# Random Forest - gini
rfc = RandomForestClassifier(n_estimators=10,criterion='gini')

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('-----------------')
print('RFC, GINI')
print(cm)















































