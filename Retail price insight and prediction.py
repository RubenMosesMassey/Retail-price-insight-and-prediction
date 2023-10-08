#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

data = pd.read_csv("C:/Users/Ruben Moses Massey/Downloads/archive (2)/retail_price.csv")
print(data.head())


# In[6]:


print(data.isnull().sum())


# In[7]:


print(data.describe())


# In[10]:


#EDA
fig = px.histogram(data, 
                   x='total_price', 
                   nbins=20, 
                   title='Distribution of Total Price')
fig.show()


# In[11]:


fig = px.box(data, 
             y='unit_price', 
             title='Box Plot of Unit Price')
fig.show()


# In[21]:


#Relationship between quantity and total prices

fig = px.scatter(data, 
                 x='qty', 
                 y='total_price', 
                 title='Quantity vs Total Price', trendline="ols")
fig.show()


# In[22]:


fig = px.bar(data, x='product_category_name', 
             y='total_price', 
             title='Average Total Price by Product Category')
fig.show()


# In[23]:


fig = px.box(data, x='weekday', 
             y='total_price', 
             title='Box Plot of Total Price by Weekday')
fig.show()


# In[24]:


# Correlation between the numerical features with each other


# In[25]:


correlation_matrix = data.corr()
fig = go.Figure(go.Heatmap(x=correlation_matrix.columns, 
                           y=correlation_matrix.columns, 
                           z=correlation_matrix.values))
fig.update_layout(title='Correlation Heatmap of Numerical Features')
fig.show()


# In[26]:


# Training Machine Learning model for this project


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
data['comp_price_diff'] = data['unit_price'] - data['comp_1']
X = data[['qty', 'unit_price', 'comp_1', 
          'product_score', 'comp_price_diff']]
y = data['total_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=42)

# Train a linear regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[30]:


y_pred = model.predict(X_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                         marker=dict(color='blue'), 
                         name='Predicted vs. Actual Retail Price'))
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                         mode='lines', 
                         marker=dict(color='red'), 
                         name='Ideal Prediction'))
fig.update_layout(
    title='Predicted vs. Actual Retail Price',
    xaxis_title='Actual Retail Price',
    yaxis_title='Predicted Retail Price'
)
fig.show()


# In[ ]:




