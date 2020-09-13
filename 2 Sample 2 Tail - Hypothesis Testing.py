#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats


# # Question No. - 1

# In[6]:


cutlets = pd.read_csv('cutlets.csv')


# In[9]:


cutlets.describe()


# In[14]:


unit_a = cutlets['Unit A']
unit_b = cutlets['Unit B']


# In[16]:


unit_a


# In[18]:


unit_b


# In[20]:


stats.ttest_ind(unit_a, unit_b)


# In[ ]:





# In[ ]:





# # Question No. - 2

# In[24]:


labtat = pd.read_csv('LabTAT.csv')


# In[25]:


labtat


# In[28]:


lab_1 = labtat['Laboratory 1']
lab_2 = labtat['Laboratory 2']
lab_3 = labtat['Laboratory 3']
lab_4 = labtat['Laboratory 4']


# In[29]:


lab_1


# In[30]:


lab_2


# In[31]:


lab_3


# In[32]:


lab_4


# In[41]:


stats.ttest_ind(lab_1, lab_2)


# In[37]:


stats.ttest_ind(lab_1, lab_3)


# In[35]:


stats.ttest_ind(lab_1, lab_4)


# In[ ]:





# In[ ]:





# # Question No.- 3

# In[42]:


buyer = pd.read_csv('BuyerRatio.csv')


# In[43]:


buyer


# In[65]:


east_ratio = (buyer['East'][0])/(buyer['East'][1])
west_ratio = (buyer['West'][0]) / (buyer['West'][1])
north_ratio = (buyer['North'][0]) / (buyer['North'][1])
south_ratio = (buyer['South'][0]) / (buyer['South'][1])


# In[127]:


n1 = 435+50
p1 = 50/485 

n2 = 1523+142
p2 = 142/1665

n3 = 1356+131
p3 = 131/1487

n4 = 750+70
p4 = 70/820


# In[128]:


east = np.random.binomial(1, p1, n1)
west = np.random.binomial(1, p2, n2)
north = np.random.binomial(1, p3, n3)
south = np.random.binomial(1, p4, n4)


# In[132]:


stats.ttest_ind(east, west)


# In[133]:


stats.ttest_ind(east, north)


# In[134]:


stats.ttest_ind(east, south)


# In[ ]:





# In[ ]:





# # Question No. - 4

# In[74]:


data = pd.read_csv('Costomer+OrderForm.csv')


# In[75]:


data


# In[94]:


new_data = data.replace(('Error Free','Defective') , (1,0) )


# In[95]:


new_data


# In[98]:


phil_data = new_data['Phillippines']
indo_data = new_data['Indonesia']
mal_data = new_data['Malta']
ind_data = new_data['India']


# In[99]:


stats.ttest_ind(phil_data, indo_data)


# In[100]:


stats.ttest_ind(phil_data, mal_data)


# In[101]:


stats.ttest_ind(phil_data, ind_data)


# In[ ]:





# In[ ]:





# # Question No. - 5

# In[103]:


fantaloons = pd.read_csv('Faltoons.csv')


# In[104]:


fantaloons


# In[105]:


fant_new = fantaloons.replace(('Male', 'Female'), (0, 1))


# In[106]:


fant_new


# In[108]:


fant_weekday = fant_new['Weekdays']
fant_weekend = fant_new['Weekend']


# In[109]:


fant_weekday


# In[110]:


fant_weekend


# In[111]:


stats.ttest_ind(fant_weekday, fant_weekend)


# In[ ]:




