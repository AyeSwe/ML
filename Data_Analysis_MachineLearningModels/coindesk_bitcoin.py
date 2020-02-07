# ----------------------------------------------------------------------
# Name:     coindesk_bitcoin
# Purpose:  This program read the streamed cleaned coindesk API stock prices for coindesk.csv
#           perform the necessary further data preprocessing analyst with
#           Linear regression model , and predict the 1 month worth of price
#           perform the pickling for the model perform the required Visual
#           presentation for further analysis
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import datetime
import pickle


df = pd.read_csv('./Data/coindesk.csv')
df = df.set_index('Date')

style.use('ggplot')


df.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df.drop(["a"], axis=1, inplace=True)

forecast_col = 'Close'

forecast_out = int(math.ceil(0.1* len(df)))# 1 % of the data

# preparaing for the empty labels for the incoming forcast

df['label']= df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)



x = np.array(df.drop(['Close'],1)) # all columns, other than label column
y = np.array(df['Close'])# only label column


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)



# Pickling

#  with open ('./Data/inearregressionCoinDeskFitted.pickle', 'wb') as f:
#  pickle.dump(linear_regression,f)


pickled = open('./Data/linearregressionCoinDeskFitted.pickle','rb')
linear_regression = pickle.load(pickled)




accuracy = linear_regression.score(x_test,y_test)

print ('\nLinear_regression accuracy is :', accuracy,"\n")



X = x[:-forecast_out]

X_lately = x[-forecast_out:]


Forecast_set = linear_regression.predict(X_lately)


df['Forecast'] = np.nan

last_date = df.iloc[-1].name

# print ("last date is: " , last_date)
last_date = time.mktime(datetime.datetime.strptime(last_date,"%Y-%m-%d").timetuple())

one_day = 86400
next_unix = last_date + 86400



label_arry = np.array(df['label'])


next_date_array= []
# # just visualization ( later get inside the
for i in Forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_date = str(next_date)

    #print ("String next date is:", next_date)
    next_date= str.split(next_date," ")

    #print (("Splited string next date is:", next_date[0]))
    next_date = next_date[0]
    next_date_array.append(next_date) # just to get an arrray for later use
    next_unix +=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]




newDf = pd.DataFrame(columns=['Date','Price'])


newDf = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})

#print ("newDf is: ---->", newDf)

#------------------------------------------------------
df['Close'].plot()
df['Forecast'].plot()

plt.title("January 1st, 2016 To January 1st, 2019 bitcoin Stock at CoinDesk API price and 30 day forecast")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
