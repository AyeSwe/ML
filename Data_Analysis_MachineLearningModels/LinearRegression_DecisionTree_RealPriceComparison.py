# ----------------------------------------------------------------------
# Name:     LinearRegression_DecisionTree_RealPriceComparison
# Purpose:  This program will predict the 30 days stock close price
#            with all of the features (Open, high, low, Volume)
#            with Linear Regression, DecisionTree and Real price comparison
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as DecisonTree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set(font_scale = 1.2)

yahoo= pd.read_csv('./Data/yahoo.csv')

yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()





#split the dat to training data and testing data

x = yahoo.iloc[:,: -1]
print (x.shape)
y = yahoo.iloc[:,-1]
print (y.shape)
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 30 day


# # Linear Regression model

linearRegression_model = LinearRegression()
linearRegression_model.fit(x_train_data, y_train_data)
liner_predict=linearRegression_model.predict(x_test_data)
accurency_linear = linearRegression_model.score(x_test_data, y_test_data)


# normalize the x train data and x testing data
scale = StandardScaler()
xtrain_scale= scale.fit_transform(x_train_data)
xtest_scale = scale.transform(x_test_data)



# # Decision Tree model

D_Tree_model = DecisonTree(max_depth=2) # to avoid outfitting max_depth is controlled.

#  train with scaled x train data and y train data
D_Tree_model.fit(xtrain_scale, y_train_data)

#
tree_predit = D_Tree_model.predict(xtest_scale)
print ("D_Tree Prediction is ", tree_predit)

D_treeAccurecy = D_Tree_model.score(xtest_scale,y_test_data)


print("Confident  of LinearRegression model is: ", accurency_linear)

print("Confident  of Decision Tree Regression model is: ", D_treeAccurecy)

LinearModel_prediction = np.array(liner_predict)
TreeeModel_predition = np.array(tree_predit)
real_data = np.array(y_test_data)

plt.plot(LinearModel_prediction)
plt.plot(TreeeModel_predition)
plt.plot(real_data)
plt.title(" 30 day forecast price from Linear Regression Vs Decision Tree Regression Vs Real Price ")
graph = plt.subplot(111)
box = graph.get_position()
graph.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(["TreeModel Price", "LinearModel Price","Real Price"], loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()




