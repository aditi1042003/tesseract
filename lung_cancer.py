# -*- coding: utf-8 -*-
"""Lung_Cancer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ojvxCRkA63L0ZbS_FKkPtskH6S1dtDHA
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

path ="/content/lung.csv"
data = pd.read_csv(path)

y = data.LungCancer

data.columns

features =['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO',
       'CN', 'Disel', 'Air_EQI']

X = data[features]
X.head()

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print(rf_val_mae)

import pickle

pickle.dump(rf_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

ans=[[12.36	,15.77,	23.257118,	183.193624,	896.42,	19.620539,	0.014027,	0.199725,	0.131007]]

print(model.predict(ans))

