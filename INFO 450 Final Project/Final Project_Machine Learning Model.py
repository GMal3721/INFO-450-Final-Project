#!/usr/bin/env python
# coding: utf-8

# In[5]:


import seaborn as sns
import numpy as np
import pandas as pd                 #Load all dependencies into the notebook
import xgboost as xgb
from xgboost import XGBClassifier


# In[6]:


IMDB_data = 'Downloads/IMDB Top 250 Movies.csv'

movies_data = pd.read_csv(IMDB_data)          #Import the csv file from the dataset, and define the csv with the name "movies_data"


# In[7]:


movies_data.head(10) #Displays the top 10 highest reated films in the dataset so we can train and test the information


# In[48]:


from sklearn.preprocessing import OrdinalEncoder     #Imported an ordinal encoder from sklearn which transforms or encodes categorical data in an array of integers

X, y = movies_data.drop("rating", axis=1), movies_data[['rating']]

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()
dums = pd.get_dummies(X[cats])


X = X.drop(cats, axis = 1)      #Drops the columns in the category or "cats" within the datafram

X = pd.concat([X, dums], axis =1 )


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=None)  #Now we split the training and test data in order to proceed with our linear regression


# In[49]:


X_train #Display the split training data


# In[50]:


from sklearn.linear_model import LinearRegression     #import the linear regression component from sklearn
model = LinearRegression()
model.fit(X_train,y_train)                            # fit the model to the data that you are training, in this case the rating


# In[51]:


print(model.intercept_)    #The model intercept of movie ratings which should show on the linear regression model


# In[52]:


coeff_parameter = model.coef_  #Extracts the coefficient and assingns it to the "model.coef" variable


# In[53]:


predictions = model.predict(X_test)    #Displays the model predictions so it can be used in comparison with our own
predictions


# In[43]:


sns.regplot(y_test,predictions) #Displays the models linear regression


# As you can see from the graph above, our linear regression model is not the best in tersm of accuracy 

# In[ ]:




