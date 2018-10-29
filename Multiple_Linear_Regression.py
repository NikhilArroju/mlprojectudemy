# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:29:41 2018

@author: Nik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#This is my datapreprocessing template
dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

'''
from sklearn.preprocessing import Imputer
#This is used to deal with the missing values
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])
#X[:,1:3] = imputer.transform(X[:,1:3])
'''

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Dummy variable trap
X = X[:,1:]

'''
#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
'''

#bulding a optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]


def backward_elimination(x,sl):
    for i in range(0,len(x[0])):
        regressor_OLS = sm.OLS(endog = y,exog = x).fit()
        p_values = regressor_OLS.pvalues
        max_p_value = max(p_values)
        max_p_index = list(p_values).index(max_p_value)
        if max(regressor_OLS.pvalues)>sl:    
            x = np.delete(x,max_p_index,1)
        else:
            return x
    
X_modeled = backward_elimination(X_opt,0.05)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_modeled,y,test_size = 0.2,random_state = 42)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)



