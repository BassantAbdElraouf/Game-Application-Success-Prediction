import pandas as pd
from sklearn.model_selection import train_test_split
import functions as fn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

data_of_test = pd.read_csv("D:/6th Term/Pattern/Project/GameProject/games-regression-dataset.csv")
X_of_test = data_of_test.iloc[:, 0:17]
Y_of_test = data_of_test[['Average User Rating']]
print(X_of_test.dtypes)
print(X_of_test.nunique())

if X_of_test['User Rating Count'].nunique() != 0:
    X_of_test['User Rating Count'] = X_of_test['User Rating Count'].fillna(0)

if X_of_test['Price'].nunique() != 0:
    X_of_test['Price'] = X_of_test['Price'].fillna(0)

if X_of_test['Developer'].nunique() != 0:
    X_of_test['Developer'] = X_of_test['Developer'].fillna(X_of_test['Developer'].mode()[0])

if X_of_test['Size'].nunique() != 0:
    X_of_test['Size'] = X_of_test['Size'].fillna(0)

print(X_of_test.nunique())

lbl = pickle.load(open('LabelEncoder.sav', 'rb'))
colname = ['URL', 'Name', 'Description', 'Developer', 'Primary Genre', 'Subtitle']
for i in colname:
    X_of_test[i] = X_of_test[i].map(lambda s: '<unknown>' if s not in lbl.classes_ else s)
    lbl.classes_ = np.append(lbl.classes_, '<unknown>')
    X_of_test[i] = lbl.transform(X_of_test[i])

X_of_test = X_of_test.drop('URL', axis=1)
X_of_test = X_of_test.drop('Name', axis=1)
X_of_test = X_of_test.drop('ID', axis=1)
X_of_test = X_of_test.drop('Icon URL', axis=1)
X_of_test = X_of_test.drop('Description', axis=1)
X_of_test = X_of_test.drop('Primary Genre', axis=1)
X_of_test = X_of_test.drop('Subtitle', axis=1)

X_of_test = fn.in_app_Purchases_pros(X_of_test)
if X_of_test['In-app Purchases'].nunique() != 0:
    X_of_test['In-app Purchases'] = X_of_test['In-app Purchases'].fillna(0)

X_of_test = fn.AgeRate_pros(X_of_test)
if X_of_test['Age Rating'].nunique() != 0:
    X_of_test['Age Rating'] = X_of_test['Age Rating'].fillna(X_of_test['Age Rating'].mode()[0])

X_of_test = fn.languages_pros(X_of_test)
if X_of_test['Languages'].nunique() != 0:
    X_of_test['Languages'] = X_of_test['Languages'].fillna(0)

df_encoded = fn.genres_features_pros(X_of_test)

X_of_test = X_of_test.drop('Genres', axis=1)

X_of_test['Original Release Date'] = pd.to_datetime(X_of_test['Original Release Date'], dayfirst=True)
X_of_test['timestamp OR'] = pd.DatetimeIndex(X_of_test['Original Release Date']).astype(int)
X_of_test = X_of_test.drop('Original Release Date', axis=1)

X_of_test['Current Version Release Date'] = pd.to_datetime(X_of_test['Current Version Release Date'],
                                                           dayfirst=True)
X_of_test['timestamp CV'] = pd.DatetimeIndex(X_of_test['Current Version Release Date']).astype(int)
X_of_test = X_of_test.drop('Current Version Release Date', axis=1)

print(X_of_test.dtypes)
X_of_test = fn.feature_scaling_test(X_of_test)

X_of_test = pd.concat([X_of_test, df_encoded], axis=1)

print(X_of_test.nunique())

top_feature_test = []
with open('TopFeatures.pkl', 'rb') as f:
    top_feature_test = pickle.load(f)

X_of_test = X_of_test[top_feature_test]

LR = pickle.load(open('LR_Model.sav', 'rb'))

pre1 = LR.predict(X_of_test)
print("test score LR : ")
print(LR.score(X_of_test, Y_of_test))
print("test mean squared error LR : ")
print(metrics.mean_squared_error(Y_of_test, pre1))

model = pickle.load(open('Ridge_Model.sav', 'rb'))

pre3 = model.predict(X_of_test)
print("test score Ridge : ")
print(model.score(X_of_test, Y_of_test))
print("test mean squared error Ridge : ")
print(metrics.mean_squared_error(Y_of_test, pre3))

dt = pickle.load(open('DecisionTreeRegressor_Model.sav', 'rb'))

yhat = dt.predict(X_of_test)
print("test score Decision tree : ")
print(dt.score(X_of_test, Y_of_test))
print("test mean squared error Decision tree : ")
print(metrics.mean_squared_error(Y_of_test, yhat))
