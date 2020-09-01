#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install and then import matplotlib
import matplotlib.pyplot as plt


# In[2]:


#matplotlib to render plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


x = [-3, 5, 7]


# In[4]:


x


# In[5]:


y = [10, 2, 5]


# In[6]:


fig = plt.figure(figsize=(15,3))

plt.plot(x, y)
plt.xlim(-4, 10)
plt.ylim(0, 12)
plt.xlabel('X Axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Sales Comaprison', size=20, y=1.03)


# In[7]:


#fig.savefig('example.png', dpi=300)


# #### How to change the plot size 

# In[8]:


fig.get_size_inches()


# In[9]:


fig.set_size_inches(14, 4)


# In[10]:


fig


# In[11]:


fig, axs = plt.subplots(nrows=1, ncols=1)


# In[12]:


# More than one Axes with plt.subplots, then the second item in the tuple is a NumPy array containing all the Axes
fig, axs = plt.subplots(2, 4)


# #### Matplotlib

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


mtcars = pd.read_csv("mtcars.csv")


# In[15]:


mtcars.columns


# In[16]:


mtcars.shape


# In[17]:


mtcars.head()


# In[18]:


# table 
pd.crosstab(mtcars.gear,mtcars.cyl)


# In[19]:


# bar plot between 2 different categories 
pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar")


# In[20]:


mtcars["gear"].value_counts()


# In[21]:


mtcars["gear"].value_counts()
mtcars.gear.value_counts().plot(kind="pie")


# In[22]:


import seaborn as sns 
# getting boxplot of mpg with respect to each category of gears 
sns.boxplot(x="gear",y="mpg",data=mtcars)


# In[23]:


sns.pairplot(mtcars.iloc[:,0:4]) # histogram of each column and 
# scatter plot of each variable with respect to other columns 


# In[24]:


plt.scatter(mtcars.mpg,mtcars.qsec)## scatter plot of two variables


# In[25]:


# Graphical Representation of data
#import matplotlib.pyplot as plt
# Histogram
plt.hist(mtcars['mpg']) 


# In[26]:


plt.hist(mtcars['mpg'],facecolor ="peru",edgecolor ="blue",bins =5)
#creates histogram with 5bins and colours filled init.


# In[27]:


#Boxplot
#help(plt.boxplot)
plt.boxplot(mtcars['mpg'],vert = True)


# In[28]:


plt.boxplot(mtcars['mpg'],vert =False);plt.ylabel("MPG");plt.xlabel("Boxplot");plt.title("Boxplot")  # for vertical


# In[29]:


#Violin Plot
#help(plt.violinplot)
plt.violinplot(mtcars["mpg"])


# #### Visualizing data with pandas

# # seaborn

# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[31]:


tips =sns.load_dataset('tips')


# In[32]:


tips


# In[33]:


sns.set_style("darkgrid", {'axes.grid' : True})
sns.lmplot(x= 'total_bill', y='tip', data=tips)
plt.show()


# In[34]:


sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex',palette='Set2')
plt.show()


# In[35]:


sns.lmplot(x='total_bill', y='tip', data=tips, col='sex')
plt.show()


# ## Univariate → “one variable” data visualization

# In[36]:


#strip plot

sns.stripplot(y= 'tip', data=tips,jitter=False)
plt.ylabel('tip ($)')
plt.show()


# In[37]:


tips.head()


# In[38]:


#Grouping with stripplot()
sns.stripplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()


# In[39]:


sns.stripplot(x='day', y='tip', data=tips, size=4,jitter=False)
plt.ylabel('tip ($)')
plt.show()


# In[40]:


#Swarm plot

sns.swarmplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()
    


# In[41]:


sns.swarmplot(x='day', y='tip', data=tips, hue='sex')
plt.ylabel('tip ($)')
plt.show()


# In[42]:


#Changing orientation
sns.swarmplot(x='tip', y='day', data=tips, hue='sex',orient='h')
plt.xlabel('tip ($)')
plt.show()


# ### Box and Violin plot

# In[43]:


#plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')


# In[44]:


#plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

plt.show()


# In[45]:


plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

plt.show()


# In[46]:


##Combining plots

sns.violinplot(x='day', y='tip', data=tips, inner=None,color='lightgray')
sns.stripplot(x='day', y='tip', data=tips, size=4,jitter=True)
plt.ylabel('tip ($)')
plt.show()


# ## Bivariate → “two variables” data visualization

# In[47]:


#Joint plot
sns.jointplot(x= 'total_bill', y= 'tip', data=tips)
plt.show()


# In[48]:


# Density plot
sns.jointplot(x='total_bill', y= 'tip', data=tips,kind='kde')
plt.show()


# In[49]:


#Pair plot

sns.pairplot(tips)
plt.show()


# In[50]:


sns.pairplot(tips, hue='sex')
plt.show()


# In[51]:


tips.corr()


# In[52]:


#Covariance heat map of tips data
sns.heatmap(tips.corr(),vmin=-1, vmax=1, cmap='ocean')


# # Titanic data visualization

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# ### Loading dataset

# In[55]:


data = pd.read_csv("train.csv")
data.head()


# 
# #survival - Survival (0 = No; 1 = Yes)
# class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# name - Name
# sex - Sex
# age - Age
# sibsp - Number of Siblings/Spouses Aboard
# parch - Number of Parents/Children Aboard
# ticket - Ticket Number
# fare - Passenger Fare
# cabin - Cabin
# embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# ### Data types

# In[56]:


print(data.dtypes)


# We can see that numbers are represted as int or float in this dataset and data type conversion is not needed here.

# ### Proportion of target (Survived)

# In[57]:


data.groupby('Survived')['PassengerId'].count()


# This dataset has a decent proportion of target class and it is not skewed to any one.

# ### Visual Exploration

# In[58]:


f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x='Survived', y="Age",  data=data);


# In[59]:


f, ax = plt.subplots(figsize=(7,7))
sns.barplot(x='Sex', y="Survived",  data=data);


# In[60]:


f, ax = plt.subplots(figsize=(11,5))
sns.boxplot(x="Sex", y="Age", hue="Survived", data=data);


# In[61]:


sns.barplot(x="Pclass", y="Survived", data=data);


# In[62]:


sns.barplot(x="Pclass", y="Survived",hue="Sex", data=data);


# In[63]:


sns.barplot(x="SibSp", y="Survived", data=data);


# In[64]:


sns.barplot(x="Parch", y="Survived", data=data);


# In[65]:


survived = data.loc[data['Survived']==1,"Age"].dropna()
sns.distplot(survived)
plt.title("Survived");


# In[66]:


not_survived = data.loc[data['Survived']==0,"Age"].dropna()
sns.distplot(not_survived)
plt.title("Not Survived");


# Infants had high survival rate and elderly passengers above 65+ were less likely to survive

# In[67]:


sns.pairplot(data.dropna());


# In[68]:


# Pclass vs Survive
grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.4, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[69]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5,ci=None)
grid.add_legend()


# In[70]:


data.columns


# In[71]:


grid = sns.FacetGrid(data, row='Pclass', col='Sex', )
grid.map(plt.hist, 'Age', alpha=.5, bins=40)
grid.add_legend()


# In[72]:



sns.distplot(a=data[data['Embarked']=='C']['Survived'],bins=3,kde=False)
plt.title("Cherbourg")
plt.xticks([0,1])
plt.show()
plt.title("QueensTown")
sns.distplot(a=data[data['Embarked']=='Q']['Survived'],bins=3,kde=False)
plt.xticks([0,1])

plt.show()
plt.title("Southampton")
sns.distplot(a=data[data['Embarked']=='S']['Survived'],bins=3,kde=False)
plt.xticks([0,1])

plt.show()
#Most of the Passengers embarked from Southampton


# In[73]:


#Plotting correlation by using heatmap
sns.heatmap(data.corr(),cmap='CMRmap')
plt.legend()


# In[74]:


figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
data.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
data.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
data.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
data.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
data.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=data,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=data,ax=axesbi[1,2])


# In[ ]:




