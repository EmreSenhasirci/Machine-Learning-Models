5# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:33:31 2023

@author: casper
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
veriler = pd.read_csv('satislar.csv')
aylar = veriler.iloc[:,0:1]


satislar = veriler.iloc[:,1:2]

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
x_train = x_train.sort_index()
y_train = y_train.sort_index()
x_test = x_test.sort_index()
y_test = y_test.sort_index()
lr = LinearRegression()

lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)
plt.plot(x_train,y_train)
plt.plot(x_test, tahmin)
print(tahmin)
