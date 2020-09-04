import numpy as np
import pandas as pd

import matplotlib.pyplot as plt #импорт библиотек

from sklearn.datasets import load_diabetes #загрузка тренировочных данных
dia_data = load_diabetes()
print(dia_data['DESCR']) #вывод информации о тренировочных данных

X = pd.DataFrame(dia_data['data'], columns=dia_data['feature_names'])
y = dia_data['target']

X.hist(X.columns, figsize=(10, 10)); #вывод тренировочных данных визуально

import seaborn as sns

plt.figure(figsize=(10,7))
sns.heatmap(X.corr())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7) #разделение тренировочных и тестовых данных

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression #линейная регрессия

model = LinearRegression()

model.fit(X_train, y_train)

y_train_prediction = model.predict(X_train)
y_test_prediction = model.predict(X_test)

plt.figure(figsize=(20, 8)) 
plt.bar(X.columns, model.coef_)  #итоговый вывод

from sklearn.metrics import mean_squared_error, mean_absolute_error #поиск ошибок

print(f'Тренировочная относительная ошибка: {mean_squared_error(y_train, y_train_prediction)}')
print(f'Тестовая относительная ошибка: {mean_squared_error(y_test, y_test_prediction)}')

print(f'Тренировочная абсолютная ошибка: {mean_absolute_error(y_train, y_train_prediction)}')
print(f'Тестовая абсолютная ошибка: {mean_absolute_error(y_test, y_test_prediction)}')

plt.show()