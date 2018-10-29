# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:38:55 2018

@author: Nik
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#This is my datapreprocessing template
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

'''
from sklearn.preprocessing import Imputer
#This is used to deal with the missing values
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
X[:,0] = imputer.fit_transform(X[:,0])
#X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#NO NEED 'CAUSE THE LINEAR REGRESSION LIBRARY TAKES CARE OF IT
#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
'''

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)




