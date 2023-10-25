#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('iris.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df['Species'].value_counts()


# In[8]:


df.isnull().sum()


# In[10]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Species',y='SepalLengthCm',data=df.sort_values('SepalLengthCm',ascending=False))


# In[11]:


df.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm')


# In[12]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=5)


# In[13]:


sns.pairplot(df, hue="Species", size=3)


# In[14]:


df.boxplot(by="Species", figsize=(12, 6))


# In[16]:


import pandas.plotting
from pandas.plotting import andrews_curves
andrews_curves(df, "Species")


# In[17]:


plt.figure(figsize=(15,15))
sns.catplot(x='Species',y='SepalWidthCm',data=df.sort_values('SepalWidthCm',ascending=False),kind='boxen')


# In[18]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# In[19]:


X=df.drop('Species',axis=1)
y=df['Species']


# In[ ]:





# In[ ]:




