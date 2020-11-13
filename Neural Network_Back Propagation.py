#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install keras')


# In[ ]:


get_ipython().system('pip install tensorflow')


# In[3]:


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy


# In[2]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


# In[3]:


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# ![image.png](attachment:image.png)

# In[4]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[5]:


# Fit the model
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)


# In[6]:


# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[7]:


# Visualize training history

# list all data in history
model.history.history.keys()


# In[11]:


# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:




