#!/usr/bin/env python
# coding: utf-8

# # Parkinsons Disease Prediction Using SVM

# Importing the dependencies

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# Data Collection & Analysis

# In[3]:


#loading the data from csv file to Pandas DataFrame
parkinsons_data = pd.read_csv('parkinsons.csv')


# In[4]:


#printing the first 5 rows of the dataset
parkinsons_data.head()


# In[5]:


# number of rows and coloumns in the dataframe
parkinsons_data.shape


# In[6]:


# getting some information on dataframe
parkinsons_data.info()


# In[7]:


# Checking for missing values in each coloumn
parkinsons_data.isnull().sum()


# In[8]:


#getting some statistical measures about the data 
parkinsons_data.describe()


# In[9]:


# Distribution of target variable
parkinsons_data['status'].value_counts()


# 1 --> Parkinson's Positive
# 
# 0 --> Healthy

# In[10]:


# grouping the data based on target variable
parkinsons_data.groupby('status').mean()


# Data Pre-Processing

# Separating the features and Target

# In[11]:


X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']


# In[12]:


print(Y)


# Splitting the data into Training and Testing data
# 

# In[13]:


X_train, X_test, Y_train, Y_test= train_test_split(X , Y, test_size=0.2, random_state=2)


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# Data standardization

# In[15]:


scaler = StandardScaler()


# In[16]:


scaler.fit(X_train)


# In[17]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[18]:


print(X_train)


# Model Training

# Support Vector Machine

# In[19]:


model = svm.SVC(kernel='linear')


# In[20]:


#training the SVM model using the training the data 
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[21]:


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction) 


# In[22]:


print('The accuracy score of the taining data ',training_data_accuracy)


# In[23]:


# accuracy score on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test,X_test_prediction)


# In[24]:


print('The accuracy score of the testing data ',test_data_accuracy)


# Building a Predictive System

# In[26]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not have Parkinsons Disease")

else:
  print("The Person has Parkinsons Disease")


# In[ ]:




