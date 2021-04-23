import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import xlrd
import re

df = pd.read_csv('C:\\Users\\User\\Documents\\sentiment analysis\\coin analysis\\Ripple\\RippleData.csv' ,sep='\t',encoding='utf-8')

#Applying filter on language
df = df[(df.language =='en')]



#Now cleaning the text of tweets to get accurate sentiment.
def cleanTxt(tweet):
    tweet = re.sub(r'@[A-Za-z0-9]+', '',tweet) #remove mention
    tweet = re.sub(r'#', '',tweet) #Removing the # Symbol
    tweet = re.sub(r'RT[\s]+', '',tweet) #Removing RT
    tweet = re.sub(r'https?:\/\/\S+', '',tweet) #Remove the link
    #tweet = re.sub("#\S+", " ", tweet) #Remove hashtag
    tweet = re.sub(r'[^\w\s]', '', tweet) #Remove punctuations
    tweet = re.sub("\'\w+", '', tweet) # Remove ticks and the next character
    tweet = re.sub(r'\w*\d+\w*', '', tweet) #Remove numbers
    tweet = re.sub('\s{2,}', " ", tweet)
    #tweet = tweet.encode('ascii', 'ignore').decode()
    #tweet = ' '.join([word for word in tweet.split() if word not in (stopwords.words('english'))])
    return tweet

df['ctweet']= df['tweet'].apply(cleanTxt)


Sentiment = []


for i in df['ctweet'].values:
    try:
        analysis = TextBlob(i)

        if (analysis.sentiment.polarity == 0):
            Sentiment.append('Neutral')
        elif (analysis.sentiment.polarity < 0.00):
            Sentiment.append('Negative')
        elif (analysis.sentiment.polarity > 0.00):
            Sentiment.append('Positive')
    except:
        exit()



df['TextBlobsentiment'] = Sentiment


#TextBlob package
def getPolarity(titles):
    return TextBlob(titles).sentiment.polarity

#calculating subjectivity,Polarity and objectivity of the tweets
def getsubjectivity(tweets):
    return TextBlob(tweets).sentiment.subjectivity

df['TextblobScore'] = df['ctweet'].apply(getPolarity)
df['TSubjectivity'] = df['ctweet'].apply(getsubjectivity)


#Export Data
df.to_csv(r'C:/Users/Shanu Rinkeshwar/Desktop/mr_bee/Dogecoin/RippleData.csv', index = False)
