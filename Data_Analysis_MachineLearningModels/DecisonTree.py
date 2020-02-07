# ----------------------------------------------------------------------
# Name:     DecisionTree
# Purpose:  This program will predict the 30 days stock close price with all of the features (Open, high, low, Volume)
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(font_scale = 1.2)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as DecisonTree
from scipy import stats


yahoo= pd.read_csv('./Data/yahoo.csv')

yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)

yahoo['close']= yahoo['Close']

yahoo.drop(['Close'], axis =1, inplace= True)

yahoo = yahoo.dropna()

print ("Data summery before outlier removal: ", yahoo.describe())


# dropping any outlier that out of 3 standard deviation from the column mean (99.7%)
yahoo = yahoo[np.abs(stats.zscore(yahoo)< 2).all(axis=1)]

print (yahoo.info())

print ("Data summery before outlier removal:")
print(yahoo.describe())


#split the dat to training data and testing data

x = yahoo.iloc[:,: -1]

y = yahoo.iloc[:,-1]

x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 10 day


# normalize the x train data and x testing data
scale = StandardScaler()
xtrain_scale= scale.fit_transform(x_train_data)
xtest_scale = scale.transform(x_test_data)


#  Decision Tree model

D_Tree_model = DecisonTree(max_depth=2) # to avoid outfitting max_depth is controlled.

# train with scaled x train data and y train data
D_Tree_model.fit(xtrain_scale, y_train_data)


tree_predit = D_Tree_model.predict(xtest_scale)

#print ("D_Tree Prediction is ", tree_predit)
D_treeAccurecy = D_Tree_model.score(xtest_scale,y_test_data)


print ("Decision Tree model accuracy is: ", D_treeAccurecy )


#  Evaluation of Decision tree model with Root Mean square error


#  Model Evaluation Metric for LinearRegression
#  RMSE (Root Mean Squared error checking for features)
#print ("Root Mean Square Error is ",np.sqrt(metrics.mean_squared_error(y_test_data, tree_predit)))

tree_predit_array = np.array(tree_predit)
y_test_data_array = np.array(y_test_data)



newDf = pd.DataFrame(columns=['Close'])
newDf['Close']= tree_predit_array # fill the column with Date
newDf.to_csv('./Data/DecisionTreePrediction.csv')

plt.plot(y_test_data_array)
plt.plot(tree_predit_array )
plt.title(" 30 day forecast with Decision Tree Regression price and Real Price ")

graph = plt.subplot(111)
box = graph.get_position()
graph.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(["Predicted Price","Real Price" ], loc='center left', bbox_to_anchor=(legend_x, legend_y))
