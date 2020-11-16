#!/usr/bin/env python
# coding: utf-8

# #### Upsampling Data

# In[1]:


# upsample to daily intervals
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot


# In[3]:


series = read_csv('sales.csv', header=0, index_col=0, parse_dates=True,squeeze=True)


# In[4]:


series


# In[5]:


upsampled = series.resample('D').mean()
print(upsampled.head(32))


# ##### interpolate the missing value

# In[6]:


interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
pyplot.show()


# #### Downsampling Data

# In[7]:


# downsample to quarterly intervals
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot


# In[9]:


resample = series.resample('Q')
quarterly_mean_sales = resample.mean()
quarterly_mean_sales


# # Tranformations

# In[10]:


# load and plot a time series
from pandas import read_csv
from matplotlib import pyplot


# In[11]:


series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True)


# In[12]:


# line plot
pyplot.subplot(211)
pyplot.plot(series)
# histogram
pyplot.subplot(212)
pyplot.hist(series)
pyplot.show()


# #### Square Root Transform

# In[14]:


from pandas import read_csv
from pandas import DataFrame
from numpy import sqrt
from matplotlib import pyplot


# In[15]:


dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = sqrt(dataframe['passengers'])


# In[16]:


# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()


# #### Log Transform

# In[17]:


from numpy import log
dataframe = DataFrame(series.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = log(dataframe['passengers'])

# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])
pyplot.show()


# In[ ]:





# In[ ]:


print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()

