# ----------------------------------------------------------------------
# Name:     Read_Json_File
# Purpose:  This program will read tweeter data jason file and gather the required data
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

import datetime
import pandas as pd
import json

tweets_data_path = 'tweet_data_testing.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")

# reading line by line from json file
for line in tweets_file:
    try:
        tweet = json.loads(line)

        # changing Twitter_API time format to python yy,m,d
        tweet_daytime = datetime.datetime.fromtimestamp(int(tweet['timestamp_ms']) / 1000)
        tweet_day = tweet_daytime.strftime('%Y-%m-%d')
        tweets_data.append(tweet)
    except:
        continue

tweets = pd.DataFrame(tweets_data)

# this replace the Date Column of the tweets with converted Date format
tweets['Date'] = tweet_day

print(" tweets['Date']:is  ", tweets['Date'])
print(tweets.info())







