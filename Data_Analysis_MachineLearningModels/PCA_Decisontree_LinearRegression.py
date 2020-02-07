# ----------------------------------------------------------------------
# Name:     PCA_Decisiontree_LinearRegression
# Purpose:  This program will reduce the dimension of yahoo finance
#           6 columns into two columns as Price and other features, to analyse if the
#           the behavior of the Linear regression, and svm model
#           dropped Adj Close because it is the same with Close column
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------


"""

"""
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas_datareader.data as web
from matplotlib import style
from sklearn.tree import DecisionTreeRegressor as DTreeRegree
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set(font_scale = 1.2)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

yahoo= pd.read_csv('./Data/yahoo.csv')
# testing for dropping date column, Adj Close column
#df_Forcast.drop(["a"], axis=1, inplace=True)
yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()
print ("Statistic summary of the data")
print (yahoo.describe())

# outlier removal with 2 standard variation

#yahoo = yahoo[np.abs(stats.zscore(yahoo)< 2).all(axis=1)] # this outliter remove increase the root mean square error.

x = yahoo.iloc[:,: -1]# evey row and every column other than the last column
#print (x.shape)
y = yahoo.iloc[:,-1]# only the last column : this is class data
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 10 day

PCA = PCA()
TreeModel= DTreeRegree()
LinearModel = LinearRegression()

X = PCA.fit_transform(x_train_data)
TreeModel.fit(X, y_train_data)

LinearModel.fit(X,y_train_data)

X_testData_PCA_fit = PCA.transform(x_test_data)
#print (X_testData_PCA_fit)
Tree_predit = TreeModel.predict(X_testData_PCA_fit)

Linear_predit = LinearModel.predict((X_testData_PCA_fit))

Tree_score = cross_val_score(TreeModel, x_train_data, y_train_data)
Linear_score = cross_val_score(LinearModel, x_train_data, y_train_data)

print("The prediction score of PCA_Decison Tree Ressor is: ", Tree_score)
print("The prediction score of PCA_Linear Regression is: ", Linear_score)

print ("Root Mean Square Error for Decision Tree is ", np.sqrt(metrics.mean_squared_error(y_test_data, Tree_predit)))
print ("Root Mean Square Error for linear Regression is ", np.sqrt(metrics.mean_squared_error(y_test_data, Linear_predit)))

