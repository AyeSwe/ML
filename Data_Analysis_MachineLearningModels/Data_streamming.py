# ----------------------------------------------------------------------
# Name:     Data_streamming
# Purpose:  This program live stream the bitcoin price from the coindesk API
#           3 years of daily stock information are gathered
#           perform some necessary data preparation before saving to the further analysis
#           streamed data are saved as coindesk.csv
#
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

import requests
import datetime as dt
import pandas as pd
import pandas_datareader.data as web


r = requests.get('https://api.coindesk.com/v1/bpi/historical/close.json?start=2016-01-01&end=2019-01-02').json()


#sending jason file into the dat frame
df = pd.DataFrame(r, columns=['bpi'])

# drop the null values
df.dropna(inplace=True)


# sending values of the bpi as price
Price = df['bpi'].values


# sending key (Dates) of the bpi as Date
Date= df['bpi'].keys()


# making a new data frame with columns labels
newDf = pd.DataFrame(columns=['Date','Close'])
newDf['Date']= Date # fill the column with Date
newDf['Close']= Price# fill the column with Prices

newDf.to_csv('./Data/coindesk.csv')

#----------------------Stramming of coinDesk data end here and yahoo finiciance data streamming start here----------------------


symbol= 'BTC-USD'

start= dt.datetime(2016,1,1)
end = dt.datetime.now()
df = web.DataReader(symbol, 'yahoo', start, end)
df.to_csv('./Data/yahoo.csv')

df = pd.read_csv('./Data/yahoo.csv',parse_dates=True,index_col=0 )
df = df.round(4)
df.to_csv('./Data/yahoo.csv')

