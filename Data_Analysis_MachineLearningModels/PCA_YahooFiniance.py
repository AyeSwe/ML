# ----------------------------------------------------------------------
# Name:     PCA_YahooFiniance
# Purpose:  This program will reduce the dimension of yahoo finance 6 columns
#           into two columns as Price and other features, to analyse if the
#           the behavior of the Linear regression, and svm model
#           dropped Adj Close because it is the same with Close column
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set(font_scale = 1.2)


df = pd.read_csv('./Data/yahoo.csv')
df = df.set_index('Date')
print (df.head())

df.drop(["Adj Close"], axis=1, inplace=True)

print (df.head())


#-----------Data Preparation Section start herer -------------

# seperating the label column and named as label

label = df.iloc[:,3]
label = pd.DataFrame(label)
label.to_csv('./Data/PCA_label.csv')

df['Volume']= df.Volume.astype(float)
print (label)

# separating the features columns in one dataframe
features = df.drop(df.columns[3],axis = 1)

print ("features are --->",features)
print (features.info())


# standarized the all features in the dataset

standard_scaler = StandardScaler()
standard_scaler.fit(features)

# this will transform to array
transformed_data = standard_scaler.transform(features)
print(transformed_data)

# do the matrix Transform
Transformed_matrix = features.T

# # here finding the covariance matrix for the Eigen Vectors and Values
c_matrix = np.cov(Transformed_matrix)


E_values, E_vector = np.linalg.eig(c_matrix)

#print ("Eigen Values \n", E_values)


# # got max eign value
max_Eigne = E_values.max()
# get the percentage variance of the max Eigen value
sum_all_Eignen_values= sum(E_values)


PC1_variance_percentage = max_Eigne/sum_all_Eignen_values
PC1_variance_percentage=np.round(PC1_variance_percentage* 100, 1)
print ("\nPC1 variance percentage is ====>",PC1_variance_percentage, "%")# wow that is very big percentage PC1

# get PC2 percentage
second_max_eigne= E_values[1]


PC2_variance_percentage = second_max_eigne/sum_all_Eignen_values
PC2_variance_percentage=np.round(PC2_variance_percentage* 100, 1)
print ("PC2 variance percentage is ====> ",PC2_variance_percentage, "%")
#
# #now project the data point to the PC1
#
PC1 = features.dot(E_vector.T[0])
PC2 = features.dot((E_vector.T[1]))
PC1.to_csv('./Data/PC1.csv') # for later K_mean use, will save this file
#

#---------visualization-----------

s= 5
plt.scatter(PC1,label, s, c="g", marker='d',
             label="Closing Price")
plt.xlabel("PC1")
plt.ylabel("Prices")
plt.title("PC1 features vs Closing Price")
plt.legend(loc='upper left')
plt.show()



