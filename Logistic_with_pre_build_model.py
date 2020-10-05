#!/usr/bin/env python
# coding: utf-8

# In[2]:







import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load


# In[3]:


st.title('Model Deployment: Logistic Regression')


# In[4]:


st.sidebar.header('User Input Parameters')


# In[5]:


def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR = st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT = st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS = st.sidebar.number_input("Insert Loss")
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features = pd.DataFrame(data,index = [0])
    return features


# In[6]:


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# In[7]:


# load the model from disk
loaded_model = load(open('Logistic_Regression_streamlit.sav', 'rb'))


# In[8]:


prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)


# In[9]:


st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')


# In[10]:


st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




