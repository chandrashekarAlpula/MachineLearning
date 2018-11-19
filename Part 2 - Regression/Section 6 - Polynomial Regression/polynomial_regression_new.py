# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:14:06 2018

@author: chandra.shekhar
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X,y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 3)
X_poly = polyReg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, c='red')
plt.plot(X, linReg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
plt.scatter(X,y,c='red')
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)), c='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X1 = X
X1 = np.append(arr=X1, values=np.ones((1,1)).astype(float), axis=0)
X1[10,:] = [6.5]
y1 = linReg2.predict(polyReg.fit_transform(X1[10:11,0:1] ))

linReg.predict(6.5)
linReg2.predict(polyReg.fit_transform(6.5))