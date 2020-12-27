#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re


# In[2]:


laptop_reviews=[]


# In[3]:


### Extracting reviews from Amazon website ################
for i in range(1,10):
    laptop=[]  
    url="https://www.amazon.in/ask/questions/asin/B08HYXNRSL/2/ref=ask_dp_iaw_ql_hza?isAnswered=true#question-Tx1EB59RLIX240R"+str(i)
    response = requests.get(url)
    soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
    reviews = soup.findAll("span",attrs={"class","a-size-base review-text"})# Extracting the content under specific tags  
    for i in range(len(reviews)):
        laptop.append(reviews[i].text)  
    laptop_reviews=laptop_reviews+laptop  # adding the reviews of one page to empty list which in future contains all the reviews


# In[8]:


with open("laptop.txt","w",encoding='utf8') as output:
    output.write(str(laptop_reviews))


# In[10]:


lp_rev_string = " ".join(laptop_reviews)


# In[11]:


lp_rev_string = re.sub("[^A-Za-z" "]+"," ",lp_rev_string).lower()
lp_rev_string = re.sub("[0-9" "]+"," ", lp_rev_string)


# In[12]:


lp_rev_words = lp_rev_string.split(" ")


# In[13]:


with open("stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


# In[15]:


lp_rev_string = " ".join(lp_rev_words)


# In[17]:


with open("positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")
    
poswords = poswords[36:]  


# In[18]:


with open("negative-words.txt","r") as neg:
    negwords = neg.read().split("\n")

negwords = negwords[37:]


# In[19]:


cleaned_reviews= re.sub('[^A-Za-z0-9" "]+', '', lp_rev_string)


# In[20]:


f = open("review.txt","w")


# In[23]:


f.write(cleaned_reviews)
f.close()


# In[25]:


with open("The_Post.text","w") as fp:
    fp.write(str(cleaned_reviews))


# In[ ]:




