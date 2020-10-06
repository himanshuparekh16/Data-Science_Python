#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load


# In[2]:


st.title('Model Deployment: Logistic Regression')


# In[10]:


df = pd.read_csv('Claimants_Test.csv')
df.drop(["CASENUM"],inplace=True,axis = 1)
df = df.dropna().reset_index()
df.drop(["index"],inplace=True,axis = 1)


# In[11]:


#st.subheader('User Input parameters')
st.write(df)


# In[12]:


# load the model from disk
loaded_model = load(open('Logistic_Model.sav', 'rb'))


# In[13]:


prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)


# In[14]:


#st.subheader('Predicted Result')
#st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')


# In[15]:


st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[16]:


output=pd.concat([df,pd.DataFrame(prediction_proba)],axis=1)

output.to_csv('output.csv')


# In[ ]:




