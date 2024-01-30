#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


pwd


# In[25]:


pip install --upgrade jupyter


# In[26]:


variable=(r"C:\Users\Navya\task - 1\titanic.csv")
variable.head()


# In[27]:


variable.shape


# In[28]:


variable.describe()
variable['Survived'].value_counts()


# In[29]:


sns.countplot(x=variable['Survived'],hue=variable['Pclass'])
variable["Sex"]
            


# In[32]:


sns.countplot(x=variable["Sex"],hue=variable['Survived'])


# In[34]:


variable.groupby('Sex')[['Survived']].mean()
variable['Sex'].unique()


# In[36]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

variable = pd.read_csv(r"C:\Users\Navya\task - 1\titanic.csv")
variable['Sex']=LabelEncoder().fit_transform(variable['Sex'])
variable.head()


# In[37]:


variable['Sex'],variable['Survived']


# In[47]:


variable=variable.drop(['Age'],axis=1)



# In[38]:


import seaborn as sns
sns.countplot(x=variable['Sex'],hue=variable['Survived'])

variable.isna().sum()


# In[50]:


print(variable.head())


# In[56]:


variable_final=variable


# In[57]:


variable_final.head(10)


# In[64]:


x=variable[['Pclass','Sex']]
y=variable['Survived']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression

log=LogisticRegression(random_state=0)
log.fit(x_train,y_train)


pred=print(log.predict(x_test))

print(y_test)


# In[66]:


import warnings
warnings.filterwarnings("ignore")
res=log.predict([[34,1]])
if(res == 0):
    print("Sorry! Not Survived");
else:
    print("Congratulation! Survived");


# In[ ]:




