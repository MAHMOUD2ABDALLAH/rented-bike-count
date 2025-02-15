# Rented Bike Count Regression Report

This report outlines the steps taken to preprocess data, analyze features, and apply various regression techniques to predict bike rental counts. The goal is to understand the impact of different features and models on the prediction accuracy.

## Preprocessing Techniques

### 1. Handling Categorical Features
- **Categorical Features:** Season, Holiday, Functioning Day
- **Encoding:** Converted categorical features to numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder
encoding = LabelEncoder()
bike['Seasons'] = encoding.fit_transform(bike['Seasons'])
bike['Holiday'] = encoding.fit_transform(bike['Holiday'])
bike['Functioning Day'] = encoding.fit_transform(bike['Functioning Day'])
```
- **Season Adjustment:** Added an autumn season to ensure all four seasons are represented.

### 2. Scaling Numerical Features
- **Features with High Ranges:** Visibility (10m), Temperature (°C), Dew Point Temperature (°C)
- **Scaling:** Applied `MinMaxScaler` to scale features to a range of (0,1).

```python
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
bike['Temperature (°C)'] = scale.fit_transform(bike[['Temperature (°C)']])
bike['Dew point temperature (°C)'] = scale.fit_transform(bike[['Dew point temperature (°C)']])
bike['Solar Radiation (MJ/m2)'] = scale.fit_transform(bike[['Solar Radiation (MJ/m2)']])
bike['Humidity(%)'] = scale.fit_transform(bike[['Humidity(%)']])
bike['Wind speed (m/s)'] = scale.fit_transform(bike[['Wind speed (m/s)']])
```

### 3. Handling Missing Values
- **Imputation:** Used `SimpleImputer` to replace missing values with the mean of the column.

```python
from sklearn.impute import SimpleImputer
import numpy as np
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
bike = imputer.fit_transform(bike)
```

## Feature Selection
- **Key Features Identified:** Hour, Temperature (°C), Solar Radiation (MJ/m2)
- **Method Used:** `SelectFromModel` with `RandomForestRegressor`

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
select2 = SelectFromModel(RandomForestRegressor())
selected = select2.fit_transform(x, y)
print(selected.shape)
print(select2.get_support())
```
- **Feature Importance Results:**
  - Features selected: Hour, Temperature (°C), Solar Radiation (MJ/m2)
  - When only these features were used, a high mean square error (~648.11) was observed, indicating that more features need to be included.

## Data Splitting (Training, Testing, Validation)

- **Splitting ratios explored:** (80:20), (90:10), (60:40), (50:50)
- **Final choice:** Default parameters (75:25)

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33)
```

## Regression Techniques

### 1. Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
LR = LinearRegression().fit(x_train, y_train)
y_predict = LR.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print('Mean Square Error:', np.sqrt(mse))
```

### 2. Ridge Regression
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
y_predict = ridge.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print('Mean Square Error:', np.sqrt(mse))
```

### 3. Lasso Regression
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
y_predict = lasso.predict(x_train)
mse = mean_squared_error(y_train, y_predict)
print('Mean Square Error:', np.sqrt(mse))
```

### 4. Elastic Net Regression
```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.05)
elastic_net.fit(x_train, y_train)
y_predict = elastic_net.predict(x_test)
mse = mean_squared_error(y_test, y_predict)
print('Mean Square Error:', np.sqrt(mse))
```

### 5. Ensemble Models

#### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_predict = model.predict(x_train)
mse = mean_squared_error(y_train, y_predict)
print('Mean Square Error:', np.sqrt(mse))
```

#### Gradient Boosting Regressor
```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(learning_rate=0.05)
model.fit(x_train, y_train)
y_predict = model.predict(x_train)
mse = mean_squared_error(y_train, y_predict)
print('Mean Square Error:', np.sqrt(mse))
```

## Model Comparison
| Model | MSE |
|--------|----------|
| Linear Regression | 342.16 |
| Lasso Regression | 428.54 |
| Ridge Regression | 427.86 |
| Elastic Net | 439.25 |
| Random Forest Regressor | 86.56 |
| Gradient Boosting Regressor | 223.37 |

## Conclusion

- **Best Performing Model:** Random Forest Regressor (Lowest MSE: 86.56)
- **Feature Importance:** Hour, Temperature (°C), and Solar Radiation (MJ/m2) were the most impactful features.
- **Scaling:** `MinMaxScaler` was effective in normalizing the data.
- **Final Thoughts:** Data scaling and feature selection helped improve performance, but including more features in the model yielded better results than strict feature selection.

## Data Visualization
A function was created to visualize the difference between predicted and actual values.

```python
def plotGraph(y_train, y_pred_train, rand):
    import matplotlib.pyplot as plt
    if max(y_train) >= max(y_pred_train):
        my_range = int(max(y_train))
    else:
        my_range = int(max(y_pred_train))
    plt.scatter(range(len(y_train)), y_train, color='blue')
    plt.scatter(range(len(y_pred_train)), y_pred_train, color='red')
    plt.title(rand)
    plt.show()
```
<img width="392" alt="VIS" src="https://github.com/user-attachments/assets/59943f95-5150-41e1-be29-2f130562ca86" />


This graph helped in analyzing how well the model performed in comparison to actual data.

---
This report provides a comprehensive overview of the preprocessing steps, feature analysis, and regression techniques applied to the bike rental dataset. The **Random Forest Regressor** emerged as the most effective model for this prediction task.

