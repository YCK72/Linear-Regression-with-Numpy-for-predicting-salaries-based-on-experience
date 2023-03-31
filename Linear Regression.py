#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Numpy is a python library which is used to perform a wide variety of
#mathematical operations on arrays.

#matplotlib.pyplot is a collection of functions that make matplotlib
#work like MATLAB

#Pandas are used for working with different data sets. Its major function
#are analyzing, cleaning, exploring and manipulatind data.


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salery_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlable('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




