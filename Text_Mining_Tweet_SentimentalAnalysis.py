#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tweepy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re


# In[2]:


# Fill the X's with the credentials obtained by  
# following the above mentioned procedure. 
consumer_key = "ouCsWbPYBmpwgwg4kbXFFYGlL" 
consumer_secret = "T49DJqZVUJ4whYz8WoB9AfQcNnBSXuPAq64pq45InxuVdpAg6E"
access_key = "851823646154805248-fSg8mp9w6SxB3lqjlG5VDqTxqV8OYbt"
access_secret = "rTCBCkDmakwqU0QE7ZjQvJmJIoNoQjlG3xgehyhlkJEpu"


# In[3]:


alltweets = []


# In[4]:


def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    
    oldest = alltweets[-1].id - 1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        #save most recent tweets
        alltweets.extend(new_tweets)
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))                # tweet.get('user', {}).get('location', {})
 
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    
    import pandas as pd
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv(screen_name+"_tweets.csv")
    return tweets_df


# In[5]:


DonaldTrump = get_all_tweets("realDonaldTrump")


# ## Sentimental Analysis

# In[6]:


with open("Trump.txt","w",encoding='utf8') as output:
    output.write(str(DonaldTrump))


# In[7]:


# Joinining all the reviews into single paragraph 
DT_tweet_string = " ".join(DonaldTrump)


# In[8]:


# Removing unwanted symbols incase if exists
DT_tweet_string = re.sub("[^A-Za-z" "]+"," ",DT_tweet_string).lower()
DT_tweet_string = re.sub("[0-9" "]+"," ",DT_tweet_string)


# In[9]:


DT_tweet_words = DT_tweet_string.split(" ")


# In[10]:


with open("stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


# In[11]:


DT_tweet_string = " ".join(DT_tweet_words)


# In[12]:


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(DT_tweet_string)

plt.imshow(wordcloud_ip)


# In[13]:


with open("positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")
    
poswords = poswords[36:]          


# In[14]:


with open("negative-words.txt","r") as neg:
    negwords = neg.read().split("\n")

negwords = negwords[37:]


# In[15]:


cleaned_tweets= re.sub('[^A-Za-z0-9" "]+', '', DT_tweet_string)


# In[16]:


f = open("tweet.txt","w")
f.write(cleaned_tweets)
f.close()


# In[ ]:




