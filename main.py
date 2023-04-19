import pandas as pd
import numpy as np
import csv
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import float64, int64
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# label encoding for categorical vars
bike = pd.read_csv("G:\\regression\\train.csv")
encoding = LabelEncoder()
bike['Seasons'] = encoding.fit_transform(bike['Seasons'])
bike['Holiday'] = encoding.fit_transform(bike['Holiday'])
bike['Functioning Day'] = encoding.fit_transform(bike['Functioning Day'])

# preprocessing MinMax scaler
scale = MinMaxScaler(copy=True, feature_range=(0, 1))
x1 = np.array(bike['Temperature(°C)'], dtype=float64)
bike['Temperature(°C)'] = scale.fit_transform(x1.reshape(-1, 1))
x2 = np.array(bike['Dew point temperature(°C)'], dtype=float64)
bike['Dew point temperature(°C)'] = scale.fit_transform(x2.reshape(-1, 1))
x3 = np.array(bike['Solar Radiation (MJ/m2)'], dtype=float64)
bike['Solar Radiation (MJ/m2)'] = scale.fit_transform(x3.reshape(-1, 1))
x4 = np.array(bike['Humidity(%)'], dtype=int64)
bike['Humidity(%)'] = scale.fit_transform(x4.reshape(-1, 1))
x5 = np.array(bike['Wind speed (m/s)'], dtype=float64)
bike['Wind speed (m/s)'] = scale.fit_transform(x5.reshape(-1, 1))
x6 = np.array(bike['Visibility (10m)'], dtype=int64)
bike['Visibility (10m)'] = scale.fit_transform(x6.reshape(-1, 1))
x7 = np.array(bike['Hour'], dtype=int64)
bike['Hour'] = scale.fit_transform(x7.reshape(-1, 1))
# # preprocessing standardization
# # scale = StandardScaler(copy=True, with_mean=True, with_std=True)
# # xs = scale.fit_transform(xs.reshape(-1, 1))
#
# # preprocessing MinMaxscaler
# scale2 = MinMaxScaler(copy=True, feature_range=(0, 1))
# xs = scale2.fit_transform(xs.reshape(-1, 1))
#
# # preprocessing Normalizer
# # scale3 = Normalizer(norm='l2', copy=True)
# # xs = scale3.fit_transform(xs.reshape(-1, 1))
#
# # preprocessing MaxAbsscaler
# # scale3 = MaxAbsScaler(copy=True)
# # xs = scale3.fit_transform(xs.reshape(-1, 1))
# # create train and test data


# data cleaning for the missed values
imputes = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(bike)

x = bike.iloc[:, 1:]
y = bike['Rented Bike Count']
print(x.head())
print(y.head())
# ==========================================================================

# feature selection from model
select2 = SelectFromModel(RandomForestRegressor())
Selected = select2.fit_transform(x, y)
print(Selected.shape)
print(select2.get_support())

# dataset splitting to training and testing data
# x_train takes 75% inputs
# x_test takes 25% inputs
# y_train takes 75% actual values of outputs of x_train
# y_test takes 25% actual values of outputs of x_test for prediction
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=20)

# ==============================MODELS=====================================
# # 1st Linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_train)
print('predict values:', y_predict)
mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
print('mean square error:', np.sqrt(mse))
print('accurecy:', accuracy_score(y_train, y_predict))
# ==========================================================================
# 2nd sklearn module for lasso regression

# lasso = Lasso(alpha=1.0)
# lasso.fit(x_train, y_train)
# y_predict = lasso.predict(x_train)
# print('predict values:', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))
# ===========================================================================
# 3rd sklearn module for  ridge regression
# ridge = Ridge(alpha=0.1)
# ridge.fit(x_train, y_train)
# y_predict = ridge.predict(x_train)
# print('predict values:', y_predict)
# mse = mean_squared_error(y_train, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))
# ============================================================================
# 4th sklearn module for elastic net regression
# [elastic_net = ElasticNet(alpha=0.05)
# elastic_net.fit(x_train, y_train)
# y_predict = elastic_net.predict(x_test)
# print('predict values:', y_predict)
# print('actual values:', y_test)
# mse = mean_squared_error(y_test, y_predict, multioutput='uniform_average')
# print('mean square error:', np.sqrt(mse))]
# ==================================ENSEMBLE===========================================
# 5th sklearn module for random forest model
# model = RandomForestRegressor()
# model.fit(x_train, y_train)
# y_predict_train = model.predict(x_train)
# print('predict values :', y_predict_train)
# mse = mean_squared_error(y_train, y_predict_train, multioutput='uniform_average')
# print('mean square error :', np.sqrt(mse))
# =====================================================================================
# 6th sklearn module for gredient boosting regressor model
model = GradientBoostingRegressor(learning_rate=0.05)
model.fit(x_train, y_train)
y_predict_train = model.predict(x_train)
print('predict values :', y_predict_train)
mse = mean_squared_error(y_train, y_predict_train, multioutput='uniform_average')
print('mean square error :', np.sqrt(mse))


# =================================PRINTING CSV============================================
# testx = pd.read_csv('G:\\regression\\test.csv')
# testx['Seasons'] = encoding.fit_transform(testx['Seasons'])
# testx['Holiday'] = encoding.fit_transform(testx['Holiday'])
# testx['Functioning Day'] = encoding.fit_transform(testx['Functioning Day'])
#
# scale2 = MinMaxScaler(copy=True, feature_range=(0, 1))
# x11 = np.array(testx['Temperature(°C)'], dtype=float64)
# testx['Temperature(°C)'] = scale2.fit_transform(x11.reshape(-1, 1))
# x22 = np.array(testx['Dew point temperature(°C)'], dtype=float64)
# testx['Dew point temperature(°C)'] = scale2.fit_transform(x22.reshape(-1, 1))
# x44 = np.array(testx['Solar Radiation (MJ/m2)'], dtype=float64)
# testx['Solar Radiation (MJ/m2)'] = scale2.fit_transform(x44.reshape(-1, 1))
# x55 = np.array(testx['Humidity(%)'], dtype=int64)
# testx['Humidity(%)'] = scale2.fit_transform(x55.reshape(-1, 1))
# x66 = np.array(testx['Wind speed (m/s)'], dtype=float64)
# testx['Wind speed (m/s)'] = scale2.fit_transform(x66.reshape(-1, 1))
# x77 = np.array(testx['Visibility (10m)'], dtype=int64)
# testx['Visibility (10m)'] = scale2.fit_transform(x77.reshape(-1, 1))
# x88 = np.array(testx['Hour'], dtype=int64)
# testx['Hour'] = scale2.fit_transform(x88.reshape(-1, 1))
# # data cleaning
# imputes2 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(testx)
#
# testdataid = testx['ID'].values
# x2 = testx.iloc[:, 1:]
# y_pred2 = model.predict(x2)
# final = pd.DataFrame({'ID': testdataid, 'Rented Bike Count': y_pred2})
# print(final)
# final.to_csv('G:\\prediction.csv')
# print('sucess')


# =======================================GRAPH==============================================
def plotGraph(y_train, y_pred_train, rand):
    if max(y_train) >= max(y_pred_train):
        my_range = int(max(y_train))
    else:
        my_range = int(max(y_pred_train))
    plt.scatter(range(len(y_train)), y_train, color='blue')
    plt.scatter(range(len(y_pred_train)), y_pred_train, color='red')
    plt.title(rand)
    plt.show()
    return


plotGraph(y_train, y_predict_train, 'random forest')
