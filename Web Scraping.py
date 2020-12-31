#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd


# In[2]:


driver = webdriver.Chrome("D:\Data Science - ExcelR\Data Science Project\chromedriver")


# In[3]:


products=[] #List to store name of the product
prices=[] #List to store price of the product
ratings=[] #List to store rating of the product


# In[4]:


driver.get("https://www.flipkart.com/computers/laptops/~acer-gaming-laptops/pr?sid=6bo,b5g&wid=3.productCard.PMU_V2_2")


# In[5]:


content = driver.page_source
soup = BeautifulSoup(content)

for a in soup.findAll('a',href=True, attrs={'class':'E2-pcE _1q8tSL'}):
    name=a.find('div', attrs={'class':'_4rR01T'})
    price=a.find('div', attrs={'class':'_30jeq3 _1_WHN1'})
    rating=a.find('div', attrs={'class':'_3LWZlK'})
    products.append(name.text)
    prices.append(price.text)
    ratings.append(rating.text)


# In[6]:


df = pd.DataFrame({'Product Name':products,'Price':prices,'Rating':ratings}) 


# In[7]:


df.to_csv('products.csv', index=False, encoding='utf-8')


# In[8]:


df


# In[ ]:




