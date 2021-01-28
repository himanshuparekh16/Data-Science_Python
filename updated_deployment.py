#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd


# In[ ]:





# In[2]:


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# In[ ]:





# In[3]:


st.markdown(
    """
    <style>
    .reportview-container {
        background: #8cd7b3
    }
   .sidebar .sidebar-content {
        background: #8dc5bf
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[ ]:





# In[4]:


df=pd.read_csv("Final_Data.csv")


# In[ ]:





# In[5]:


st.write("""

# Drug Analysis

""")


# In[ ]:





# In[6]:


drug_name = df['drugName']
condition = df['condition']
side_effect = df['Side_Effect']
effectiveness = df['effectiveness']


# In[ ]:





# In[7]:


dummy = df['drugName'].tolist()


# In[ ]:





# In[13]:


st.sidebar.text(" ")
st.sidebar.text(" ")
drug = st.sidebar.selectbox('Drug Name:' , options=drug_name)
st.sidebar.text(" ")
st.sidebar.text(" ")


# In[ ]:





# In[9]:


index = dummy.index(drug)


# In[ ]:





# In[11]:


if(st.sidebar.button('Show Result')):
    
    st.info("Information")
    
    st.markdown("<h3  style= 'color: blue';>Side Effects</h3>", unsafe_allow_html=True)
    a = side_effect[index]
    st.write(a)
    st.text(" ")
    
    st.markdown("<h3  style= 'color: blue';>Conditions</h3>", unsafe_allow_html=True)
    b = condition[index]
    st.write(b)
    st.text(" ")
    
    st.markdown("<h3  style= 'color: blue';>Effectiveness</h3>", unsafe_allow_html=True)
    c = effectiveness[index]
    st.write(c)
    st.text(" ")
    
    
    
else:
    st.text(" ")
    st.text(" ")
    st.text(" ")
    
    st.markdown("<h3  style= 'color: blue';>Side Effects</h3>", unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")
    st.text(" ")
    
    st.markdown("<h3  style= 'color: blue';>Conditions</h3>", unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")
    st.text(" ")
    
    st.markdown("<h3  style= 'color: blue';>Effectiveness</h3>", unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")
    st.text(" ")
    


# In[ ]:




