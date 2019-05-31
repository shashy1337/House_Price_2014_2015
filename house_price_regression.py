#!/usr/bin/env python
# coding: utf-8

# # House Sale Prediction 2014-2015

# # 1.Import, cleaning data and visualization

# Import libraries

# In[8]:


import numpy as np
import pprint
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
from IPython.display import display
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks.offline
import cufflinks as cf
cufflinks.offline.run_from_ipython()
sns.set(style="ticks")
cf.set_config_file(world_readable=True, theme='pearl', offline=True)
plt.rc('font', family='Verdana')


# Import data and transform in to pandas dataframe 

# In[9]:


data = pd.read_csv('/home/shashy/Загрузки/housesalesprediction/kc_house_data.csv')
data.head()


# Check the size of dataset

# In[46]:


data.shape


# Check the info about dataset and search a null objects

# In[11]:


data.info()


# Some visualization of price

# In[12]:


data['price'].iplot(kind='hist', xTitle='price',
                  yTitle='count', title='Price')


# Heatmap

# In[13]:


f, ax = plt.subplots(figsize=(16, 12))
corr = data.corr()
sns.heatmap(corr, annot=True)


# Visualization price of the houses versus footage ofthe home by date

# In[14]:


data.iplot(
    x='sqft_living',
    y='price',
    # Указываем категорию
    categories='date',
    xTitle='square footage of the home',
    yTitle='price of the houses',
    title='Price of The Houses vs Square Footage Of The Home by Date ')


# delete unnecessary columns

# In[15]:


Clear_data_X = data.drop(columns=['id', 'date'], axis=1)
Clear_data_X.head()


# Create a discrette scatter to show the relationship between the data

# In[16]:


sns.pairplot(Clear_data_X)


# # 2.Splitting + Normalizing data and Predicted Models

# Separate the target variable from the master data and divide the data into test and training samples.
# 
# 

# In[17]:


X = Clear_data_X
y = data['price'].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# After the data is divided, it is necessary to normalize them with the help of a scaler

# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])


# We start training our models. For training, I chose 3 models: 1.LinearRegression, 2.RidgeRegression and 3.RandomForrestRegression. As presented below, the linear models and the ensemble of solutions to a random forest coped very well with their work and made a fairly accurate forecast based on test data.

# In[20]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True).fit(X_train, y_train)
y_head_lr = lr.predict(X_test)


# LinearRegression - 0.99 * 100 = 99% prediction

# In[21]:


print("R^2 on train Linear Regression: {:.2f}".format(lr.score(X_train, y_train)))
print("R^2 on test Linear Regression: {:.2f}".format(lr.score(X_test, y_test)))


# In[22]:


print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(lr.predict(X_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(lr.predict(X_test.iloc[[2],:])))


# RidgeRegression (with params: alpha = 10) - 0.99 * 100 = 99% prediction

# In[23]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 10).fit(X_train, y_train)


# In[24]:


print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(ridge.predict(X_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(ridge.predict(X_test.iloc[[2],:])))


# In[25]:


from sklearn.metrics import r2_score
y_head_ridge = ridge.predict(X_test)
r2score1 = r2_score(y_test, y_head_ridge)
print('R^2 score on ridge: {:.2f}'.format(r2score1))


# RandomForrestRegressor (with params: n_jobs = -1, n_estimators = 10, verbose = 3) - 0.99 * 100 = 99% predict

# In[30]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, n_estimators=10, verbose=3)
rf.fit(X_train,y_train)
y_head_rf = rf.predict(X_test)


# In[36]:


print("real value of y_test[1]: " + str(y_test[1]) + " -> the predict: " + str(rf.predict(X_test.iloc[[1],:])))
print("real value of y_test[2]: " + str(y_test[2]) + " -> the predict: " + str(rf.predict(X_test.iloc[[2],:])))


# In[37]:


print("R^2 score of RandomForrestRegression is: {:.2f}".format(r2_score(y_test, y_head_rf)))


# And below is the final accuracy chart of all 3 models.

# In[49]:


f, ax = plt.subplots(figsize=(16,10))
y = np.array([r2_score(y_test,y_head_lr),r2_score(y_test,y_head_ridge), r2_score(y_test, y_head_rf)])
x = ["LinearRegression","RidgeReg.", "RandomForrestReg"]
plt.bar(x,y)
plt.title("Accuracy of the models")


# Also, I saved the first 100 parameters of each model in the lists. From these lists, you can safely withdraw any value from 1-100

# In[43]:


settings = range(1, 101)
lr_predict = []
ridge_predict = []
rf_predict = []
for n_settings in settings:
    lr_predict.append("real value of LinearRegression on y_test: " + str(y_test[n_settings]) + " -> the predict: " + str(lr.predict(X_test.iloc[[n_settings],:])))
    ridge_predict.append("real value of RidgeRegression on y_test: " + str(y_test[n_settings]) + " -> the predict: " + str(ridge.predict(X_test.iloc[[n_settings],:])))
    rf_predict.append("real value of RandomForrestREgression on y_test: " + str(y_test[n_settings]) + " -> the predict: " + str(rf.predict(X_test.iloc[[n_settings],:])))


# In[50]:


print(lr_predict[1])


# # 3.Conclusion 

# Thus, we obtained fairly clear machine learning models with an accuracy of 99%. They made a visualization, cleared and divided the data into training and test samples and then normalized the data.
# 
# 
