# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Aumatic deletion of featuer based on BackwardEliminaton logic
def BackwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
onh = OneHotEncoder(categorical_features=[3])
X = onh.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
#Regression models has implicit removal of dummy variable but doing explicitly as part of notes
X = X[:,1:] ## Removing the first dummy variable(index 0) to avoid Dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 
## NO need to do feature scaling for multiple regression as library takes care of it
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()

linReg.fit(X_train, y_train)

y_pred = linReg.predict(X_test)


#Building an optimal model using Backward ELemenation
import statsmodels.formula.api as sm
#add extra 1's columns(bias)
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()