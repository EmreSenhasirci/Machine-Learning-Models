# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 23:25:29 2023

@author: casper
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# Reading Data and defining dependent and independent
df = pd.read_csv('maaslar.csv')

x = df.iloc[:,1:2]
X = x.values
y = df.iloc[:,2:]
Y = y.values

# Polynomial Regression Part
from sklearn.preprocessing import PolynomialFeatures


lr = LinearRegression()
poly_reg = PolynomialFeatures(degree=2)
lr2 = LinearRegression()
x_poly = poly_reg.fit_transform(X)
lr2.fit(x_poly,y)
plt.title('polynomial')
plt.scatter(X,Y,color='green')
plt.plot(X,lr2.predict(x_poly),color='red')
plt.show()
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

# Support Vector Regression
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)
plt.title('SVR')
plt.scatter(x_olcekli,y_olcekli, color='blue')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color='blue')
plt.show()
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
plt.scatter(X,Y,color='green')
plt.plot(X,r_dt.predict(X),color='red')
plt.title('Decision Tree')
plt.show()

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
plt.title('Random Forest')
plt.scatter(X,Y)
plt.plot(X,rf_reg.predict(X))
print(rf_reg.predict([[20]]))
print('---------------')
from sklearn.metrics import r2_score
# Calculating Metrics with rsquare
print('Random Forest R2')
print(r2_score(Y,rf_reg.predict(X)))
print('---------------')
print('SVR R2')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))
print('---------------')
print('Decision Tree R2')
print(r2_score(Y,r_dt.predict(X)))



















































