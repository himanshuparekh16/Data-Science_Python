#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


diamonds = pd.read_csv("diamonds.csv")


# In[4]:


diamonds.columns


# In[5]:


diamonds = diamonds.head(100)


# In[6]:


pd.crosstab(diamonds.cut , diamonds.color)


# ### bar plot between 2 different categories 

# In[7]:


pd.crosstab(diamonds.cut , diamonds.color).plot(kind = 'bar')


# In[8]:


diamonds["cut"].value_counts()
diamonds.cut.value_counts().plot(kind='pie')


# In[9]:


sns.boxplot(x='cut', y='carat', data=diamonds)


# In[ ]:





# ### histogram of each column and scatter plot of each variable with respect to other columns

# In[10]:


sns.pairplot(diamonds.iloc[:, 4:9])


# In[ ]:





# ### scatter plot of two variables

# In[11]:


plt.scatter(diamonds.carat , diamonds.depth)


# In[ ]:





# ### Graphical Representation of data 
# ### Histogram

# In[12]:


plt.hist(diamonds.price)


# In[13]:


plt.hist(diamonds['table'],facecolor ="green",edgecolor ="yellow",bins =5)


# ### Boxplot

# In[14]:


plt.boxplot(diamonds.carat , vert = True)


# In[15]:


plt.boxplot(diamonds.price , vert = False)
plt.ylabel("Price");plt.title("Boxplot")


# In[ ]:





# ### Violin Plot

# In[16]:


plt.violinplot(diamonds.depth)


# In[ ]:





# In[ ]:





# In[17]:


sns.set_style("darkgrid", {'axes.grid' : True})
sns.lmplot(x = 'carat', y = 'price', data = diamonds)
plt.show()


# In[19]:


sns.lmplot(x='depth', y='price', data=diamonds, col='cut')
plt.show()


# In[ ]:





# ### Strip Plot

# In[20]:


sns.stripplot( y = 'carat', data=diamonds , jitter = False)
plt.ylabel('Carat')
plt.show()


# In[ ]:





# ### Grouping with stripplot()

# In[21]:


sns.stripplot(x='carat', y='price', data = diamonds)
plt.ylabel('Price ($)')
plt.show()


# In[22]:


sns.stripplot(x='carat', y='price', data = diamonds, size=4, jitter = False)
plt.ylabel('Price')
plt.show()


# In[ ]:





# In[ ]:





# ### Swarm plot

# In[23]:


sns.swarmplot(x='carat', y='table', data = diamonds)
plt.ylabel('Depth')
plt.show()


# In[24]:


sns.swarmplot(x='price', y='depth', data=diamonds, hue='cut')
plt.ylabel('Depth')
plt.show()


# In[ ]:





# ### Changing orientation

# In[25]:


sns.swarmplot(x='carat', y='depth', data = diamonds, hue='cut',orient='h')
plt.xlabel('Depth')
plt.show()


# In[ ]:





# In[ ]:





# ### Box and Violin Plot

# In[26]:


#plt.subplot(1,2,1)
sns.boxplot(x='price', y='carat', data=diamonds)
plt.ylabel('Carat')


# In[ ]:




