# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:41:43 2021

@author: sunil
"""
import streamlit as st
import pandas as pd
st.set_page_config(page_title="drug analysis", page_icon="💊" ,layout="wide",
 initial_sidebar_state="expanded")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)
local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.markdown(
    """
    <style>
    .reportview-container {
        background: #d7b38c
    }
   .sidebar .sidebar-content {
        background:blue
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: red; font-size:50px'>Drug Analysis</h1>", unsafe_allow_html=True)
df=pd.read_csv("Output.csv")
drug_name = df['drugName']
condition = df['condition']
side_effect = df['Side_Effect']
effectiveness = df['effectiveness']
dummy = df['drugName'].tolist()
drug = st.sidebar.selectbox('Drug Name:' , options=drug_name)


index = dummy.index(drug)
a = side_effect[index]
#st.write(a)
b = condition[index]
#st.write(b)
c = effectiveness[index]
#st.write(c)
submit1 = st.sidebar.button('submit')
if submit1:
    st.markdown("<h3  style= 'color: blue';>Side Effects</h3>", unsafe_allow_html=True)
    st.write(a)
    st.markdown("<h3  style= 'color: blue';>Conditions</h3>", unsafe_allow_html=True)
    st.write(b)
    st.markdown("<h3  style= 'color: blue';>Effectiveness</h3>", unsafe_allow_html=True)
    st.write(c)
else:
    st.markdown("<h3  style= 'color: blue';>Side Effects</h3>", unsafe_allow_html=True)
    st.markdown("<h3  style= 'color: blue';>Conditions</h3>", unsafe_allow_html=True)
    st.markdown("<h3  style= 'color: blue';>Effectiveness</h3>", unsafe_allow_html=True)
   
    
#uploaded_file=st.sidebar.file_uploader(label="upload ur files here")
#global df1
#if uploaded_file is not None:
 #   print(uploaded_file)
  #  print("successfully uploaded")
    
   # try:
    #    d1=pd.read_csv(uploaded_file)
    #except Exception as e:
     #   print(e)
      #  df1=pd.read_excel(uploaded_file)
#try:
 #   st.write(df1)
#except Exception as e:
 #   print(e)
  #  st.write("please upload file to the application")
#df=pd.DataFrame(columns=['SideEffects','Conditions','Effectiveness'])
#SideEffects=st.write(a)
#st.write(df)
#col1,col2=st.beta_columns(2)
#col1.success(st.write(a))
import os
import base64
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href
st.sidebar.markdown(get_binary_file_downloader_html('Final_Model.ipynb', 'Model Building'), unsafe_allow_html=True)
st.sidebar.markdown(get_binary_file_downloader_html('Output.csv', 'Final Output'), unsafe_allow_html=True)
st.sidebar.markdown(get_binary_file_downloader_html('EDA.ipynb', 'EDA'), unsafe_allow_html=True)
st.sidebar.title('Reports')
model_choice = st.sidebar.selectbox('MODELS:', ['usefulcount and rating','polarity','top 10 drugs','wordclud','dustribution of rating','popular drugs','rating', '12'])
st.set_option('deprecation.showImageFormat', False)
fig = f'{model_choice}.png'
submit = st.sidebar.button('Open')
if submit:
    st.image(open(fig, 'rb').read(), format='png')

video_file = open('sk1.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)
from PIL import Image
image = Image.open('drug1.jpg')
#import numpy as np
#df = pd.DataFrame(np.random.randn(10, 5),columns=('col %d' % i for i in range(5)))
#st.table(df)
