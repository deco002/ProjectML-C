#!/usr/bin/env python
# coding: utf-8

# ## <font color=red> Reading data and creating required columns

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a plot or figure
plt.show()

import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[ ]:





# In[ ]:


data = pd.read_csv('C:\\Users\\VINCENT ALBERT\\Downloads\\ENB2012_data.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns= ["Relative Compactness", "Surface Area", "Wall Area", "Roof Area", "Overall Height", "Orientation", 
               "Glazing Area", "Glazing Area Distribution", "Heating Load", "Cooling Load"]


# In[ ]:


data.columns


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


pd.isnull(data).sum()


# ## Adding column for overall load

# In[ ]:


data['Overall Load'] = data['Heating Load'] + data['Cooling Load']
data.head()


# In[ ]:


data.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


data.corr()


# In[ ]:


sns.kdeplot(data)


# In[46]:


sns.pairplot(data, kind="scatter", diag_kind='kde')


# In[47]:


plt.subplots(figsize=(15,12))
corr = data.corr()
sns.heatmap(corr, annot=True,cmap="YlGnBu")


# ## Trend of overall load

# In[48]:


sns.distplot(data['Overall Load'])


# In[49]:


plt.boxplot(data['Overall Load'])


# In[50]:


data['Overall Load'].quantile([.25, .50, 0.75])


# ## Adding column for classes for efficiency

# In[51]:


data['Efficiency'] = np.where(data['Overall Load']<29, 'Low', np.where(data['Overall Load']<64, 'Average', 'High'))
data.head()


# In[52]:


data.shape


# In[53]:


data['Efficiency'].value_counts()


# In[54]:


sns.countplot(x='Efficiency', data=data, order=['Low','Average','High'])


# ## <font color=red> Creating different datasets for different Y variables

# ## Data set for heating load

# In[55]:


data_heat = data.drop(columns=['Cooling Load','Overall Load','Efficiency'])


# In[56]:


data_heat.head()


# In[57]:


data_heat.shape


# ## Data set for cooling load

# In[58]:


data_cool = data.drop(columns=['Heating Load','Overall Load','Efficiency'])
data_cool.head()


# In[59]:


data_cool.shape


# ## Data set for efficiency classification

# In[60]:


data_eff = data.drop(columns=['Cooling Load','Heating Load','Overall Load'])
data_eff.head()


# ## <font color=red> Models for predicting heating load

# ## Data preprocessing

# In[61]:


X = data_heat.drop(columns=['Heating Load'])
y= data_heat['Heating Load']


# In[62]:


y.head()


# In[63]:


X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# ## Neural Network Regressor

# ### Grid Search to find the Best Parameters for Nueral Network

# In[64]:


# ## Neural Network Regressor with Parameter tuning
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV

# Build the neural network regressor function to be used in the keras-sklearn wrapper
# def build_reg(optimizer):
#    nn_reg = Sequential()
#    nn_reg.add(Dense(5, input_dim=8, kernel_initializer='uniform', activation='relu'))
#    nn_reg.add(Dense(5, kernel_initializer='uniform', activation='relu'))
#    nn_reg.add(Dense(1, activation='linear'))
    
#    nn_reg.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
#    return nn_reg

#    nn_reg = KerasRegressor(build_fn=build_reg)

# param_grid = {'batch_size': [10,16,25],
#          'epochs': [50,75, 100],
#          'optimizer': ['rmsprop','adam']}

# nn_reg_grid_heat = GridSearchCV(estimator=nn_reg, param_grid=param_grid, scoring='r2', cv=8)

# nn_reg_grid_heat.fit(X_train, y_train)


# ### Best Parameters : - Batch Size= 10 , Epochs= 100, Optimizer= 'rmsprop'

# In[65]:


from keras.models import Sequential
from keras.layers import Dense

nn_reg = Sequential()
nn_reg.add(Dense(5, input_dim=8, kernel_initializer='uniform', activation='relu'))
nn_reg.add(Dense(5, kernel_initializer='uniform', activation='relu'))
nn_reg.add(Dense(1, activation='linear'))

nn_reg.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

nn_reg.fit(X_train, y_train, batch_size=10, epochs=100)


# In[ ]:


from sklearn.metrics import r2_score

y_train_predict = nn_reg.predict(X_train)
y_test_predict = nn_reg.predict(X_test)

print('Train r2: {:.2f}'.format(r2_score(y_train, y_train_predict)))
print('Test r2: {:.2f}'.format(r2_score(y_test, y_test_predict)))


# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
y_predict = lr.predict(X_train)
print('Accuracy of Linear Regression on training set: {:.2f}'
     .format(lr.score(X_train, y_train)))
print('Accuracy of Linear Regression on test set: {:.2f}'
     .format(lr.score(X_test, y_test)))


# In[ ]:


print("lr.coef_: {}".format(lr.coef_))

print("lr.intercept_: {}".format(lr.intercept_))


# ## KNN Regressor

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [2,4,6,8,10,12]}
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=8).fit(X_train, y_train)
print('Accuracy of Knn Regressor on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of Knn Regressor on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# ## Ridge

# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train,y_train)
param_grid = {'alpha':[0.01, 1, 5, 10, 100]}
grid_search = GridSearchCV(ridge, param_grid, cv = 5)
grid_search.fit(X_train, y_train)
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


ridge = Ridge(alpha=0.01).fit(X_train, y_train)
print('Accuracy of Ridge Regressor on training set: {:.2f}'
     .format(ridge.score(X_train, y_train)))
print('Accuracy of Ridge Regressor on test set: {:.2f}'
     .format(ridge.score(X_test, y_test)))


# ## Lasso

# In[ ]:


from sklearn.linear_model import Lasso
lasso = Lasso(max_iter = 10000).fit(X_train,y_train)
param_grid = {'alpha':[0.01, 1, 5, 10, 100]}
grid_search = GridSearchCV(lasso, param_grid, cv = 5)
grid_search.fit(X_train, y_train)
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


lasso = Lasso(max_iter = 10000, alpha=0.01).fit(X_train,y_train)
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
print('Accuracy of Lasso Regressor on training set: {:.2f}'
     .format(lasso.score(X_train, y_train)))
print('Accuracy of Lasso Regressor on test set: {:.2f}'
     .format(lasso.score(X_test, y_test)))


# In[ ]:


print('Features with non-zero weight (sorted by absolute magnitude):')
for e in sorted (list(zip(list(X), lasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))


# ## Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
print("X Train Polynomial Shape: - {}".format(X_train_poly.shape))
print("X Test Polynomial Shape: - {}".format(X_test_poly.shape))


# In[ ]:


lr = LinearRegression().fit(X_train_poly, y_train)
predict_lr= lr.predict(X_test_poly)
print('(poly deg 2) R-squared score (training): {}'
     .format(lr.score(X_train_poly, y_train)))
print('(poly deg 2) R-squared score (test): {}\n'
     .format(lr.score(X_test_poly, y_test)))


# ##### Addition of many polynomial features often leads to overfitting, so we often use polynomial features in combination with regression that has a regularization penalty, like ridge regression

# ## Polynomial with Ridge

# In[ ]:


ridge = Ridge().fit(X_train_poly, y_train)
print('(poly deg 2 + ridge) R-squared score (training): {}'
     .format(ridge.score(X_train_poly, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {}'
     .format(ridge.score(X_test_poly, y_test)))


# ## Linear SVR

# In[ ]:


from sklearn.svm import LinearSVR
lsvr = LinearSVR(random_state=10).fit(X_train, y_train)
param_grid = {'C': [0.01,0.1,1,10,100]}
lsvr=LinearSVR()
grid_search = GridSearchCV(lsvr,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


lsvr=LinearSVR(C=10).fit(X_train, y_train)
print('Accuracy of Linear SVR classifier on training set: {:.2f}'
     .format(lsvr.score(X_train, y_train)))
print('Accuracy of Linear SVR classifier on test set: {:.2f}'
     .format(lsvr.score(X_test, y_test)))


# ## SVM Regressor

# In[ ]:


from sklearn.svm import SVR
param_grid = {'C': [0.01,0.1,1,10,100], 'epsilon': [0.01,0.1,1,10,100]}
svr = SVR()
grid_search = GridSearchCV(svr,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


svr = SVR(C=100, epsilon=1).fit(X_train, y_train)
print('Accuracy of Knn Regressor on training set: {:.2f}'
     .format(svr.score(X_train, y_train)))
print('Accuracy of Knn Regressor on test set: {:.2f}'
     .format(svr.score(X_test, y_test)))


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
param_grid = {'max_depth': [2, 4, 6 ,8, 10]}
grid_search = GridSearchCV(dt, param_grid,cv=5)
grid_search.fit(X_train,y_train)
grid_search.best_params_


# In[ ]:


dt = DecisionTreeRegressor(max_depth=8).fit(X_train, y_train)
print('Accuracy of DT Regressor on training set: {}'
     .format(dt.score(X_train, y_train)))
print('Accuracy of Dt Regressor on test set: {}'
     .format(dt.score(X_test, y_test)))


# ## <font color='Red'> Summary of R- Square Scores for Base Regression Models for Heating Load

# 1. Linear Regression : - 0.91
# 2. KNN Regressor : - 0.92
# 3. Ridge: - 0.91
# 4. Lasso : - 0.91
# 5. Polynomial : - 0.9929
# 6. Polynomial with Ridge: - 0.9362
# 7. Linear SVR : - 0.92
# 8. SVM : - 0.94
# 9. Decision Tree : - 0.9966

# ### Best Base Regression Model :- Decision Tree with R Square= 0.9966

# # <font color='green'>Ensembles

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
param_grid = {'max_features': [2,4,6,8],
            'max_depth': [2,4,6,8],
           'max_leaf_nodes':[2,4,6,8]}
rf = RandomForestRegressor(random_state=10)
grid_search = GridSearchCV(rf,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


rf = RandomForestRegressor(max_depth= 4, max_features=6, max_leaf_nodes=8, random_state=10).fit(X_train, y_train)
print('Accuracy of RF Regressor on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of Rf Regressor on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))


# ## Bagging and Pasting

# ### <font color='blue'>Bagging with KNN

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_knn = BaggingRegressor(base_estimator=knn,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging KNN Regressor on training set: {:.2f}'
     .format(bag_knn.score(X_train, y_train)))
print('Accuracy of Bagging KNN on test set: {:.2f}'
     .format(bag_knn.score(X_test, y_test)))


# ###  <font color='blue'>Pasting with KNN

# In[ ]:


paste_knn = BaggingRegressor(base_estimator=knn, max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting KNN Regressor on training set: {:.2f}'
     .format(paste_knn.score(X_train, y_train)))
print('Accuracy of Pasting KNN on test set: {:.2f}'
     .format(paste_knn.score(X_test, y_test)))


# ### <font color='blue'>Bagging with Linear SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_lsvr = BaggingRegressor(base_estimator=lsvr,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging Linear SVR Regressor on training set: {:.2f}'
     .format(bag_lsvr.score(X_train, y_train)))
print('Accuracy of Bagging Linear SVR on test set: {:.2f}'
     .format(bag_lsvr.score(X_test, y_test)))


# ### <font color='blue'>Pasting with Linear SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
paste_lsvr = BaggingRegressor(base_estimator=lsvr,max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting Linear SVR Regressor on training set: {:.2f}'
     .format(paste_lsvr.score(X_train, y_train)))
print('Accuracy of Pasting Linear SVR on test set: {:.2f}'
     .format(paste_lsvr.score(X_test, y_test)))


# ### <font color='blue'> Bagging with SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_svr = BaggingRegressor(base_estimator=svr,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging Linear SVR Regressor on training set: {:.2f}'
     .format(bag_svr.score(X_train, y_train)))
print('Accuracy of Bagging Linear SVR on test set: {:.2f}'
     .format(bag_svr.score(X_test, y_test)))


# ###  <font color='blue'>Pasting with SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
paste_svr = BaggingRegressor(base_estimator=svr,max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting Linear SVR Regressor on training set: {:.2f}'
     .format(paste_svr.score(X_train, y_train)))
print('Accuracy of Pasting Linear SVR on test set: {:.2f}'
     .format(paste_svr.score(X_test, y_test)))


# ### <font color='blue'> Bagging with Decision Tree

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_dt = BaggingRegressor(base_estimator=dt,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging Decision Tree Regressor on training set: {:.2f}'
     .format(bag_dt.score(X_train, y_train)))
print('Accuracy of Bagging Decision Tree Regressor on test set: {:.2f}'
     .format(bag_dt.score(X_test, y_test)))


# ### <font color='blue'>Pasting with Decision Tree

# In[ ]:


from sklearn.ensemble import BaggingRegressor
paste_dt = BaggingRegressor(base_estimator=dt,max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting Decision Tree on training set: {:.2f}'
     .format(paste_svr.score(X_train, y_train)))
print('Accuracy of Pasting Decision Tree on test set: {:.2f}'
     .format(paste_svr.score(X_test, y_test)))


# ## Boosting

# ### <font color='blue'> Adaptive Boosting with Decision Tree Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adaboost_dt = AdaBoostRegressor(base_estimator = dt, learning_rate = 0.5).fit(X_train, y_train)
print('Accuracy of Adaptive Boosting with Decision Tree Regressor on training set: {}'
     .format(adaboost_dt.score(X_train, y_train)))
print('Accuracy of Adaptive Boosting with Decision Tree Regressor on test set: {}'
     .format(adaboost_dt.score(X_test, y_test)))


# ### <font color='blue'> Adaptive Boosting with Random Forest Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adaboost_rf = AdaBoostRegressor(base_estimator = rf, learning_rate = 0.5).fit(X_train, y_train)
print('Accuracy of Adaptive Boosting with Random Forest Regressor on training set: {}'
     .format(adaboost_rf.score(X_train, y_train)))
print('Accuracy of Adaptive Boosting with Random Forest Regressor on test set: {}'
     .format(adaboost_rf.score(X_test, y_test)))


# ### <font color='blue'> Adaptive Boosting with SVM Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adaboost_svr = AdaBoostRegressor(base_estimator = svr, learning_rate = 0.5).fit(X_train, y_train)
print('Accuracy of Adaptive Boosting with SVM Regressor on training set: {}'
     .format(adaboost_svr.score(X_train, y_train)))
print('Accuracy of Adaptive Boosting with SVM Regressor on test set: {}'
     .format(adaboost_svr.score(X_test, y_test)))


# ## Gradient Boosting Regressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gb=GradientBoostingRegressor(max_depth=4)
param_grid = {'learning_rate': [0.1,1,10,100], 'n_estimators': [5,10,20,50]}
gb = GradientBoostingRegressor()
grid_search = GridSearchCV(gb,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


gb=GradientBoostingRegressor(learning_rate=1, n_estimators=50).fit(X_train, y_train)
print('Accuracy of GB Regressor on training set: {}'
     .format(gb.score(X_train, y_train)))
print('Accuracy of GB Regressor on test set: {}'
     .format(gb.score(X_test, y_test)))


# ## Ensemble of All Regressors- Stacking Regressor

# In[ ]:


lr=LinearRegression()
knn=KNeighborsRegressor(n_neighbors=8)
lsvr=LinearSVR(C=10)
svr = SVR(C=100, epsilon=1)
rf = RandomForestRegressor(max_depth= 4, max_features=6, max_leaf_nodes=8)


# In[ ]:


from mlxtend.regressor import StackingRegressor
stregr = StackingRegressor(regressors=[lr, knn, lsvr, svr, rf], 
                           meta_regressor=svr)
stregr.fit(X_train, y_train)
str_predict = stregr.predict(X_test)
print('Accuracy of STR Regressor on training set: {:.3f}'
     .format(stregr.score(X_train, y_train)))
print('Accuracy of STR Regressor on test set: {:.3f}'
     .format(stregr.score(X_test, y_test)))


# In[ ]:


print("Mean Squared Error: %.4f"
      % np.mean((stregr.predict(X_test) - y_test) ** 2))
print('Variance Score: %.4f' % stregr.score(X_test, y_test))


# ## <font color='Red'> Summary of R- Square Scores for Ensemble Regression Models for Heating Load

# 1. Random Forest: - 0.95
# 2. Bagging(KNN) : - 0.86
# 3. Pasting(KNN) : - 0.89
# 4. Bagging(Linear SVR) : - 0.88
# 5. Pasting(Linear SVR) : - 0.91
# 6. Bagging(SVR) : - 0.91
# 7. Pasting(SVR) : - 0.92
# 8. Bagging(Decision Tree): - 0.96
# 9. Pasting(Decision Tree): - 0.92
# 10. Adaptive Boosting (Decision Tree) : - 0.9969
# 11. Adaptive Boosting (Random Forest) : - 0.97
# 12. Adaptive Boosting (SVM Regressor) : - 0.94
# 13. Gradient Boosting : - 0.9982

# ### Best Ensemble Regression Model for Heating Load :- Gradient Boosting with R Square= 0.9982

# ## <font color=red> Models for predicting cooling load

# ## Data preprocessing

# In[ ]:


X = data_cool.drop(columns=['Cooling Load'])
y= data_cool['Cooling Load']


# In[ ]:


y.head()


# In[ ]:


X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# ## Neural Network Regressor

# ### Grid Search to find the Best Parameters for Nueral Network

# In[ ]:


# ## Neural Network Regressor with Parameter tuning
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV

# Build the neural network regressor function to be used in the keras-sklearn wrapper
# def build_reg(optimizer):
#    nn_reg = Sequential()
#    nn_reg.add(Dense(5, input_dim=8, kernel_initializer='uniform', activation='relu'))
#    nn_reg.add(Dense(5, kernel_initializer='uniform', activation='relu'))
#    nn_reg.add(Dense(1, activation='linear'))
    
#    nn_reg.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
#    return nn_reg

#    nn_reg = KerasRegressor(build_fn=build_reg)

# param_grid = {'batch_size': [10,16,25],
#          'epochs': [50,75, 100],
#          'optimizer': ['rmsprop','adam']}

# nn_reg_grid_heat = GridSearchCV(estimator=nn_reg, param_grid=param_grid, scoring='r2', cv=8)

# nn_reg_grid_heat.fit(X_train, y_train)


# ### Best Parameters : - Batch Size= 10 , Epochs= 100, Optimizer= 'rmsprop'

# In[ ]:


nn_reg = Sequential()
nn_reg.add(Dense(5, input_dim=8, kernel_initializer='uniform', activation='relu'))
nn_reg.add(Dense(5, kernel_initializer='uniform', activation='relu'))
nn_reg.add(Dense(1, activation='linear'))

nn_reg.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

nn_reg.fit(X_train, y_train, batch_size=10, epochs=100)


# In[ ]:


y_train_predict = nn_reg.predict(X_train)
y_test_predict = nn_reg.predict(X_test)

print('Train r2: {:.2f}'.format(r2_score(y_train, y_train_predict)))
print('Test r2: {:.2f}'.format(r2_score(y_test, y_test_predict)))


# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
y_predict = lr.predict(X_train)
print('Accuracy of Linear Regression on training set: {:.2f}'
     .format(lr.score(X_train, y_train)))
print('Accuracy of Linear Regression on test set: {:.2f}'
     .format(lr.score(X_test, y_test)))


# In[ ]:


print("lr.coef_: {}".format(lr.coef_))

print("lr.intercept_: {}".format(lr.intercept_))


# ## KNN Regressor

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [2,4,6,8,10,12]}
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=8).fit(X_train, y_train)
print('Accuracy of Knn Regressor on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of Knn Regressor on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


# ## Ridge

# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train,y_train)
param_grid = {'alpha':[0.01, 1, 5, 10, 100]}
grid_search = GridSearchCV(ridge, param_grid, cv = 5)
grid_search.fit(X_train, y_train)
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


ridge = Ridge(alpha=0.01).fit(X_train, y_train)
print('Accuracy of Ridge Regressor on training set: {:.2f}'
     .format(ridge.score(X_train, y_train)))
print('Accuracy of Ridge Regressor on test set: {:.2f}'
     .format(ridge.score(X_test, y_test)))


# ## Lasso

# In[ ]:


from sklearn.linear_model import Lasso
lasso = Lasso(max_iter = 10000).fit(X_train,y_train)
param_grid = {'alpha':[0.01, 1, 5, 10, 100]}
grid_search = GridSearchCV(lasso, param_grid, cv = 5)
grid_search.fit(X_train, y_train)
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


lasso = Lasso(max_iter = 10000, alpha=0.01).fit(X_train,y_train)
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
print('Accuracy of Lasso Regressor on training set: {:.2f}'
     .format(lasso.score(X_train, y_train)))
print('Accuracy of Lasso Regressor on test set: {:.2f}'
     .format(lasso.score(X_test, y_test)))


# In[ ]:


print('Features with non-zero weight (sorted by absolute magnitude):')
for e in sorted (list(zip(list(X), lasso.coef_)),
                key = lambda e: -abs(e[1])):
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))


# ## Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
print("X Train Polynomial Shape: - {}".format(X_train_poly.shape))
print("X Test Polynomial Shape: - {}".format(X_test_poly.shape))


# In[ ]:


lr = LinearRegression().fit(X_train_poly, y_train)
predict_lr= lr.predict(X_test_poly)
print('(poly deg 2) R-squared score (training): {}'
     .format(lr.score(X_train_poly, y_train)))
print('(poly deg 2) R-squared score (test): {}\n'
     .format(lr.score(X_test_poly, y_test)))


# #### Addition of many polynomial features often leads to overfitting, so we often use polynomial features in combination with regression that has a regularization penalty, like ridge regression

# ## Polynomial with Ridge

# In[ ]:


ridge = Ridge().fit(X_train_poly, y_train)
print('(poly deg 2 + ridge) R-squared score (training): {}'
     .format(ridge.score(X_train_poly, y_train)))
print('(poly deg 2 + ridge) R-squared score (test): {}'
     .format(ridge.score(X_test_poly, y_test)))


# ## Linear SVR

# In[ ]:


from sklearn.svm import LinearSVR
lsvr = LinearSVR(random_state=10).fit(X_train, y_train)
param_grid = {'C': [0.01,0.1,1,10,100]}
lsvr=LinearSVR()
grid_search = GridSearchCV(lsvr,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


lsvr=LinearSVR(C=100).fit(X_train, y_train)
print('Accuracy of Linear SVR classifier on training set: {:.2f}'
     .format(lsvr.score(X_train, y_train)))
print('Accuracy of Linear SVR classifier on test set: {:.2f}'
     .format(lsvr.score(X_test, y_test)))


# ## SVM Regressor

# In[ ]:


from sklearn.svm import SVR
param_grid = {'C': [0.01,0.1,1,10,100], 'epsilon': [0.01,0.1,1,10,100]}
svr = SVR()
grid_search = GridSearchCV(svr,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


svr = SVR(C=100, epsilon=1).fit(X_train, y_train)
print('Accuracy of Knn Regressor on training set: {:.2f}'
     .format(svr.score(X_train, y_train)))
print('Accuracy of Knn Regressor on test set: {:.2f}'
     .format(svr.score(X_test, y_test)))


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=10)
param_grid = {'max_depth': [2, 4, 6 ,8, 10]}
grid_search = GridSearchCV(dt, param_grid,cv=5)
grid_search.fit(X_train,y_train)
grid_search.best_params_


# In[ ]:


dt = DecisionTreeRegressor(max_depth=6, random_state=10).fit(X_train, y_train)
print('Accuracy of DT Regressor on training set: {}'
     .format(dt.score(X_train, y_train)))
print('Accuracy of Dt Regressor on test set: {}'
     .format(dt.score(X_test, y_test)))


# ## <font color='red'>Summary of R- Square Scores for Base Regression Models for Cooling Load

# 1. Linear Regression : - 0.89
# 2. KNN Regressor : - 0.91
# 3. Ridge: - 0.89
# 4. Lasso : - 0.89
# 5. Polynomial : - 0.9639
# 6. Polynomial with Ridge: - 0.90
# 7. Linear SVR : - 0.88
# 8. SVM : - 0.91
# 9. Decision Tree : - 0.9650

# ### Best Base Regression Model for Cooling Load :- Decision Tree with R Square= 0.9650

# # <font color='green'> Ensembles

# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
param_grid = {'max_features': [2,4,6,8],
            'max_depth': [2,4,6,8],
           'max_leaf_nodes':[2,4,6,8]}
rf = RandomForestRegressor()
grid_search = GridSearchCV(rf,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


rf = RandomForestRegressor(max_depth= 6, max_features=6, max_leaf_nodes=8).fit(X_train, y_train)
print('Accuracy of RF Regressor on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of Rf Regressor on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))


# ## Bagging and Pasting

# ### <font color='blue'> Bagging with KNN

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_knn = BaggingRegressor(base_estimator=knn,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging KNN Regressor on training set: {:.2f}'
     .format(bag_knn.score(X_train, y_train)))
print('Accuracy of Bagging KNN on test set: {:.2f}'
     .format(bag_knn.score(X_test, y_test)))


# ### <font color='blue'> Pasting with KNN

# In[ ]:


paste_knn = BaggingRegressor(base_estimator=knn, max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting KNN Regressor on training set: {:.2f}'
     .format(paste_knn.score(X_train, y_train)))
print('Accuracy of Pasting KNN on test set: {:.2f}'
     .format(paste_knn.score(X_test, y_test)))


# ### <font color='blue'> Bagging with Linear SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_lsvr = BaggingRegressor(base_estimator=lsvr,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging Linear SVR Regressor on training set: {:.2f}'
     .format(bag_lsvr.score(X_train, y_train)))
print('Accuracy of Bagging Linear SVR on test set: {:.2f}'
     .format(bag_lsvr.score(X_test, y_test)))


# ### <font color='blue'> Pasting with Linear SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
paste_lsvr = BaggingRegressor(base_estimator=lsvr,max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting Linear SVR Regressor on training set: {:.2f}'
     .format(paste_lsvr.score(X_train, y_train)))
print('Accuracy of Pasting Linear SVR on test set: {:.2f}'
     .format(paste_lsvr.score(X_test, y_test)))


# ### <font color='blue'> Bagging with SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_svr = BaggingRegressor(base_estimator=svr,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging Linear SVR Regressor on training set: {:.2f}'
     .format(bag_svr.score(X_train, y_train)))
print('Accuracy of Bagging Linear SVR on test set: {:.2f}'
     .format(bag_svr.score(X_test, y_test)))


# ### <font color='blue'> Pasting with SVR

# In[ ]:


from sklearn.ensemble import BaggingRegressor
paste_svr = BaggingRegressor(base_estimator=svr,max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting Linear SVR Regressor on training set: {:.2f}'
     .format(paste_svr.score(X_train, y_train)))
print('Accuracy of Pasting Linear SVR on test set: {:.2f}'
     .format(paste_svr.score(X_test, y_test)))


# ### <font color='blue'> Bagging with Decision Tree

# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_dt = BaggingRegressor(base_estimator=dt,bootstrap_features=True, max_samples=50).fit(X_train, y_train)
print('Accuracy of Bagging Decision Tree Regressor on training set: {:.2f}'
     .format(bag_dt.score(X_train, y_train)))
print('Accuracy of Bagging Decision Tree Regressor on test set: {:.2f}'
     .format(bag_dt.score(X_test, y_test)))


# ### <font color='blue'> Pasting with Decision Tree

# In[ ]:


from sklearn.ensemble import BaggingRegressor
paste_dt = BaggingRegressor(base_estimator=dt,max_samples=50, bootstrap=False).fit(X_train, y_train)
print('Accuracy of Pasting Decision Tree on training set: {:.2f}'
     .format(paste_svr.score(X_train, y_train)))
print('Accuracy of Pasting Decision Tree on test set: {:.2f}'
     .format(paste_svr.score(X_test, y_test)))


# ## Boosting

# ### <font color='blue'> Adaptive Boosting with Decision Tree Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adaboost_dt = AdaBoostRegressor(base_estimator = dt, learning_rate = 0.5).fit(X_train, y_train)
print('Accuracy of Adaptive Boosting with Decision Tree Regressor on training set: {}'
     .format(adaboost_dt.score(X_train, y_train)))
print('Accuracy of Adaptive Boosting with Decision Tree Regressor on test set: {}'
     .format(adaboost_dt.score(X_test, y_test)))


# ### <font color='blue'> Adaptive Boosting with Random Forest Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adaboost_rf = AdaBoostRegressor(base_estimator = rf, learning_rate = 0.5).fit(X_train, y_train)
print('Accuracy of Adaptive Boosting with Random Forest Regressor on training set: {}'
     .format(adaboost_rf.score(X_train, y_train)))
print('Accuracy of Adaptive Boosting with Random Forest Regressor on test set: {}'
     .format(adaboost_rf.score(X_test, y_test)))


# ### <font color='blue'> Adaptive Boosting with SVM Regressor

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
adaboost_svr = AdaBoostRegressor(base_estimator = svr, learning_rate = 0.5).fit(X_train, y_train)
print('Accuracy of Adaptive Boosting with SVM Regressor on training set: {}'
     .format(adaboost_svr.score(X_train, y_train)))
print('Accuracy of Adaptive Boosting with SVM Regressor on test set: {}'
     .format(adaboost_svr.score(X_test, y_test)))


# ## Gradient Boosting Regressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
gb=GradientBoostingRegressor(max_depth=4)
param_grid = {'learning_rate': [0.1,1,10,100], 'n_estimators': [5,10,20,50]}
gb = GradientBoostingRegressor()
grid_search = GridSearchCV(gb,param_grid,cv=5)
grid_search.fit(X_train, y_train)
grid_search.best_params_


# In[ ]:


gb=GradientBoostingRegressor(learning_rate=1, n_estimators=50).fit(X_train, y_train)
print('Accuracy of GB Regressor on training set: {}'
     .format(gb.score(X_train, y_train)))
print('Accuracy of GB Regressor on test set: {}'
     .format(gb.score(X_test, y_test)))


# ## Ensemble of All Regressors- Stacking Regressor

# In[ ]:


lr=LinearRegression()
knn=KNeighborsRegressor(n_neighbors=8)
lsvr=LinearSVR(C=100)
svr = SVR(C=100, epsilon=1)
rf = RandomForestRegressor(max_depth= 6, max_features=4, max_leaf_nodes=8)


# In[ ]:


from mlxtend.regressor import StackingRegressor
stregr = StackingRegressor(regressors=[lr, knn, lsvr, svr, rf], 
                           meta_regressor=svr)
stregr.fit(X_train, y_train)
str_predict = stregr.predict(X_test)
print('Accuracy of STR Regressor on training set: {:.3f}'
     .format(stregr.score(X_train, y_train)))
print('Accuracy of STR Regressor on test set: {:.3f}'
     .format(stregr.score(X_test, y_test)))


# In[ ]:


print("Mean Squared Error: %.4f"
      % np.mean((stregr.predict(X_test) - y_test) ** 2))
print('Variance Score: %.4f' % stregr.score(X_test, y_test))


# ## <font color='Red'> Summary of R- Square Scores for Ensemble Regression Models for Cooling Load

# 1. Random Forest: - 0.93
# 2. Bagging(KNN) : - 0.84
# 3. Pasting(KNN) : - 0.86
# 4. Bagging(Linear SVR) : - 0.87
# 5. Pasting(Linear SVR) : - 0.88
# 6. Bagging(SVR) : - 0.88
# 7. Pasting(SVR) : - 0.89
# 8. Bagging(Decision Tree): - 0.95
# 9. Pasting(Decision Tree): - 0.89
# 10. Adaptive Boosting (Decision Tree) : - 0.9661
# 11. Adaptive Boosting (Random Forest) : - 0.9438
# 12. Adaptive Boosting (SVM Regressor) : - 0.92
# 13. Gradient Boosting : - 0.9863

# ### Best Ensemble Regression Model for Cooling Load :- Gradient Boosting with R Square= 0.9863

# ## <font color=red> Models for classification of efficiency

# ## Data preprocessing

# In[ ]:


X = data_eff.drop(columns=['Efficiency'])
y= data_eff['Efficiency']
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
y = LE.fit_transform(y)


# In[ ]:


y


# In[ ]:


X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# ## Neural Network Classifier

# In[ ]:


y_dummy = pd.get_dummies(y)
y_dummy.head()


# In[ ]:


X_train_org, X_test_org, y_train_dummy, y_test_dummy = train_test_split(X, y_dummy, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)


# In[ ]:


nn_clf = Sequential()
nn_clf.add(Dense(5, input_dim=8, kernel_initializer='uniform', activation='relu'))
nn_clf.add(Dense(5, kernel_initializer='uniform', activation='relu'))
nn_clf.add(Dense(3, activation='sigmoid'))

nn_clf.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

nn_clf.fit(X_train, y_train_dummy, batch_size=10, epochs=150)


# In[ ]:


y_pred = nn_clf.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred


# In[ ]:


from sklearn.metrics import accuracy_score
print('Test accuracy score: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix

print('Confusion matrix for efficiency: \n')
print(confusion_matrix(y_true=y_test, y_pred=y_pred), '\n')


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print('Classification Report for efficiency: \n')
print(classification_report(y_true=y_test, y_pred=y_pred))


# In[ ]:


model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, y_train_dummy, batch_size=10, epochs=100)


# In[ ]:


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred
print('Test accuracy score: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred)))
print('Classification Report for efficiency: \n')
print(classification_report(y_true=y_test, y_pred=y_pred))


# ## Grid Search to find the Best Parameters for Nueral Network

# In[ ]:


## Neural Network Classifier with Parameter tuning
# def build_class(optimizer):
#    model = Sequential()
#    model.add(Dense(15, input_dim=8, kernel_initializer='uniform', activation='relu'))
#    model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
#    model.add(Dense(3, activation='sigmoid'))
#    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
#    return model

# model = KerasClassifier(build_fn=build_class)

# param_grid = {'batch_size': [10,16,25],
#              'epochs': [50, 100,150],
#          'optimizer': ['rmsprop','adam']}

# model = GridSearchCV(estimator=model, scoring='accuracy', cv=8, return_train_score=True, param_grid=param_grid)
# model.fit(X_train, y_train_dummy, epochs=150)


# ### Best Parameters : - Batch Size= 10 , Epochs= 100, Optimizer= 'rmsprop'

# In[ ]:


model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, y_train_dummy, batch_size=10, epochs=150)


# In[ ]:


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_pred
print('Test accuracy score: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred)))
print('Classification Report for efficiency: \n')
print(classification_report(y_true=y_test, y_pred=y_pred))


# ## <font color='green'>Accuracy of Deep Learning Model for Efficiency Classification: - 91%
