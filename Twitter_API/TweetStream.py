# ----------------------------------------------------------------------
# Name:     TweetStream
# Purpose:  This program will read required data from tweeter's API
# Author:   Aye Swe
#
# Copyright ©  Swe, Aye 2019
# ----------------------------------------------------------------------

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
import Twitter_credential # hidden file
from tweepy import OAuthHandler
from tweepy import Stream



#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print (data)
        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter_API authetification and the connection to Twitter_API Streaming API
    Listen = StdOutListener()
    auth = OAuthHandler(Twitter_credential.CONSUMER_KEY, Twitter_credential.CONSUMER_SECRET)
    auth.set_access_token(Twitter_credential.ACCESS_TOKEN, Twitter_credential.ACCESS_TOKEN_SECRET)
    stream = Stream(auth, Listen)


    stream.filter(track=['bitcoin','currency'])


