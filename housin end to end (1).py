#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_boston


# In[2]:


df=load_boston()


# In[3]:


df1=pd.DataFrame(data=df.data,columns=df.feature_names)


# In[4]:


df1


# In[5]:


print(df.DESCR)


# In[6]:


df1['Price']=df.target


# In[7]:


df1


# In[8]:


import seaborn as sns


# In[9]:


sns.heatmap(df1.corr())


# In[10]:


sns.pairplot(df1)


# In[11]:


x=df1.iloc[:,:-1]
y=df1['Price']


# In[12]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20,random_state=42)


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[14]:


xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)


# In[15]:


df.data[0]


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


Regression=LinearRegression()
Regression.fit(xtrain,ytrain)


# In[18]:


print(Regression.coef_)


# In[19]:


print(Regression.intercept_)


# In[20]:


regpred=Regression.predict(xtest)


# In[21]:


plt.scatter(ytest,regpred)


# In[22]:


##if the graph is linear it means model is good to go


# In[23]:


residuals=ytest-regpred


# In[24]:


sns.displot(residuals,kind="kde")


# In[25]:


## we should get normal distribution curve


# In[26]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(ytest,regpred))
print(mean_squared_error(ytest,regpred))


# In[27]:


import pickle


# In[28]:


with open('regmodel.pkl','wb') as file:
    pickle.dump(Regression,file)


# In[29]:


####pickle.dump(Regression,open('Regmodel.pkl'),'wb')
    


# In[30]:


pickled_model=pickle.load(open('regmodel.pkl','rb'))


# In[31]:


with open('regmodel.pkl','rb') as file:
    preg=pickle.load(file)


# In[32]:


datas=scaler.transform(df.data[0].reshape(1,-1))


# In[33]:


pickled_model.predict(datas)


# In[ ]:





# In[ ]:




