#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn
seaborn.set()


# In[2]:


# Load the data


# In[4]:


data = pd.read_csv('1.02. Multiple linear regression.csv')


# In[5]:


data


# In[6]:


# Create your first multiple linear regression


# In[7]:


data.describe()


# In[9]:


y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]


# In[11]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()


# In[12]:


results.summary()


# In[13]:


# Reviewing Adj. R-squared against the simple linear regression model
# we have added more variables but have lost value


# In[14]:


# Therefore, we drop 'Rand 1,2,3' for it is insignificant 


# In[15]:


# Having 'b0' reduced only lessens the power of the model


# In[16]:


# F test is used to measures the overall significance of a model


# In[ ]:




