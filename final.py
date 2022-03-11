#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import random as rand
import math

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


Import Data


# In[3]:


df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan%20/Regression/master/Startups_Data.csv')
df.head()


# In[4]:


df.info()


# In[5]:


plt.figure(figsize=(20, 3))

for i, col in zip(range(1, 4), df.columns):
    plt.subplot(1, 4, i)
    sns.boxplot(x=col, data=df, color='orange')
    plt.title(f"Box Plot of {col}")
    plt.tight_layout()
plt.show()


# In[6]:


plt.figure(figsize=(25, 3))

for i, col in zip(range(1, 4), df.columns):
    plt.subplot(1, 4, i)
    sns.distplot(df[col], color='red')
    plt.title(f"Distribution Plot of {col}")
    plt.tight_layout()


# In[7]:


plt.figure(figsize=(6, 3))

sns.boxplot(x='Profit', data=df, color='orange')
plt.title('Box plot of Profit')
plt.show()

plt.figure(figsize=(6, 3))

sns.distplot(df['Profit'], color='red')
plt.title('Distribution Plot of Profit')
plt.show()


# In[8]:


df[df['Profit'] < 25000]


# In[9]:


df[df['Profit'] < 100000]


# In[10]:


plt.figure(figsize=(20, 3))

for i, col in zip(range(1, 4), df.columns):
    plt.subplot(1, 4, i)
    sns.scatterplot(x=col, y='Profit', data=df)
    plt.title(f"Scatter Plot of {col} vs Profit")
    plt.tight_layout()


# In[11]:


sns.heatmap(data=df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()


# In[ ]:




