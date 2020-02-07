# ----------------------------------------------------------------------
# Name:     All_features_prediction_Yahoo
# Purpose:  This program will predict the 30 days stock close price with
#           all of the features (Open, high, low, Volume)
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set(font_scale = 1.2)
from sklearn import metrics



yahoo = pd.read_csv('./Data/yahoo.csv')
yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()
print ("Statistic summary of the data")
print (yahoo.describe())

# outlier removal with 2 standard variation

x = yahoo.iloc[:,: -1]# evey row and every column other than the last column

y = yahoo.iloc[:,-1]# only the last column : this is class data

# get the testing data out of x and y
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 10 day


# Linear Regression model
linearRegression_model = LinearRegression()
linearRegression_model.fit(x_train_data,y_train_data)


# getting the intercept point from
print ("Intercept: ", linearRegression_model.intercept_)
print ("Coefficienc", linearRegression_model.coef_)
y_prediction = linearRegression_model.predict(x_test_data)


#plotting the least Squares Line
sns.pairplot(yahoo, x_vars=['High','Low','Open','Volume'], y_vars='close', size=4, aspect=0.7, kind='reg')
plt.show()

print ('linearRegression_model.score() is :', linearRegression_model.score(x_train_data,y_train_data) )


#  Model Evaluation Metric for LinearRegression
#  RMSE (Root Mean Squared error checking for features)

print ("Root Mean Square Error is ",np.sqrt(metrics.mean_squared_error(y_test_data, y_prediction)))

realPrice = np.array(y_test_data)
newDf = pd.DataFrame(columns=['Close'])
newDf['Close']= y_prediction # fill the column with Date
newDf.to_csv('./Data/LinearRegressionPrediction.csv')


# Y prediction, and realPrice graphs
plt.plot(y_prediction)
plt.plot(realPrice)

graph = plt.subplot(111)
box = graph.get_position()
graph.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(["Real Price", "Predicted Price"], loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()









