import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from nltk import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import functions as fn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle

# shuffle=True
# train score LR : 0.09456656935515617
# train mean squared error LR : 0.5172205921515404
# test score LR : 0.09743170587932493
# test mean squared error LR : 0.49786993472159086
# train score Ridge : 0.09456656935515617
# train mean squared error Ridge : 0.5172205921515404
# test score Ridge : 0.09743170587932504
# test mean squared error Ridge : 0.49786993472159086
# ------------------------------------------------------

# train score LR : 0.09294753262702271
# train mean squared error LR : 0.5129371712624563
# train score Ridge : 0.09294753262702271
# train mean squared error Ridge : 0.5129371712624563


data = pd.read_csv("games-regression-dataset.csv")

X = data.iloc[:, 0:17]
Y = data[['Average User Rating']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,
                                                    shuffle=False)

colname = ['URL', 'Name', 'Description', 'Developer', 'Primary Genre', 'Subtitle']
# X_train = fn.label_encoding(X_train,colname )
for i in colname:
    lbl = LabelEncoder()
    lbl.fit(list(X_train[i].values))
    with open('LabelEncoder.sav', 'wb') as f:
        pickle.dump(lbl, f)

    X_test[i] = X_test[i].map(lambda s: '<unknown>' if s not in lbl.classes_ else s)
    lbl.classes_ = np.append(lbl.classes_, '<unknown>')
    X_test[i] = lbl.transform(X_test[i])
    X_train[i] = lbl.transform(list(X_train[i].values))

print('number of unique values in X_train :')
print(X_train.nunique())

X_train = X_train.drop('URL', axis=1)
X_train = X_train.drop('Name', axis=1)
X_train = X_train.drop('ID', axis=1)
X_train = X_train.drop('Icon URL', axis=1)

print('number of null values in X_train :')
print(X_train.isna().sum())

X_train = X_train.drop('Description', axis=1)

X_train = X_train.drop('Primary Genre', axis=1)

X_train = X_train.drop('Subtitle', axis=1)

X_train = fn.in_app_Purchases_pros(X_train)

X_train = fn.AgeRate_pros(X_train)

X_train = fn.languages_pros(X_train)

X_train['Genres'] = X_train['Genres'].str.split(', ')
mlb = MultiLabelBinarizer()
genre_mlb = mlb.fit_transform(X_train['Genres'])
df_encoded = pd.DataFrame(genre_mlb, columns=mlb.classes_)
X_train = X_train.drop('Genres', axis=1)

with open('GenresFeatures.pkl', 'wb') as f:
    pickle.dump(list(mlb.classes_), f)

X_train['Original Release Date'] = pd.to_datetime(X_train['Original Release Date'], dayfirst=True)
X_train['timestamp OR'] = pd.DatetimeIndex(X_train['Original Release Date']).astype('int64')
X_train = X_train.drop('Original Release Date', axis=1)
X_train['Current Version Release Date'] = pd.to_datetime(X_train['Current Version Release Date'],dayfirst=True)
X_train['timestamp CV'] = pd.DatetimeIndex(X_train['Current Version Release Date']).astype('int64')
X_train = X_train.drop('Current Version Release Date', axis=1)

"""
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)

IQR = Q3 - Q1

with open('Q1.pkl', 'wb') as f:
    pickle.dump(list(Q1), f)

with open('Q3.pkl', 'wb') as f:
    pickle.dump(list(Q3), f)

with open('IQR.pkl', 'wb') as f:
    pickle.dump(list(IQR), f)

outliers = ((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR)))
pp = ((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).sum()
print("outliers")
print(pp)

Age_mode = X_train['Age Rating'].mode().values[0]
X_train['Age Rating'] = pd.Series(np.where(((X_train['Age Rating'] < (Q1[3] - 1.5 * IQR[3]))
                                            | (X_train['Age Rating'] > (Q3[3] + 1.5 * IQR[3]))) == True,
                                           Age_mode, X_train['Age Rating']))

with open('Age_mode.pkl', 'wb') as f:
    pickle.dump(Age_mode, f)

lang_mode = X_train['Languages'].mode().values[0]
X_train['Languages'] = pd.Series(np.where(((X_train['Languages'] < (Q1[4] - 1.5 * IQR[4]))
                                           | (X_train['Languages'] > (Q3[4] + 1.5 * IQR[4]))) == True,
                                          lang_mode, X_train['Languages']))

with open('lang_mode.pkl', 'wb') as f:
    pickle.dump(lang_mode, f)

user_mode = X_train['User Rating Count'].mode().values[0]
X_train['User Rating Count'] = pd.Series(np.where(((X_train['User Rating Count'] < (Q1[0] - 1.5 * IQR[0]))
                                                   | (X_train['User Rating Count'] > (Q3[0] + 1.5 * IQR[0]))) == True,
                                                  user_mode, X_train['User Rating Count']))

with open('user_mode.pkl', 'wb') as f:
    pickle.dump(user_mode, f)

tcv_mode = X_train['timestamp CV'].median()
X_train['timestamp CV'] = pd.Series(np.where(((X_train['timestamp CV'] < (Q1[7] - 1.5 * IQR[7]))
                                              | (X_train['timestamp CV'] > (Q3[7] + 1.5 * IQR[7]))) == True,
                                             tcv_mode, X_train['timestamp CV']))

with open('tcv_mode.pkl', 'wb') as f:
    pickle.dump(tcv_mode, f)

pp = ((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).sum()
print("outliers")
print(pp)
"""
X_train = fn.feature_scaling(X_train)
X_train = pd.concat([X_train, df_encoded], axis=1)

data1 = X_train
data1['Average User Rating'] = y_train
corr = data1.corr()

top_feature = corr.index[abs(corr['Average User Rating']) > 0.02]
top_feature = top_feature.delete(-1)

with open('TopFeatures.pkl', 'wb') as f:
    pickle.dump(top_feature, f)

X_train = X_train[top_feature]

LR = LinearRegression()
LR = LR.fit(X_train, y_train)

with open('LR_Model.sav', 'wb') as f:
    pickle.dump(LR, f)

pre = LR.predict(X_train)

print("train score LR : ")
print(LR.score(X_train, y_train))
print("train mean squared error LR : ")
print(metrics.mean_squared_error(y_train, pre))

model = Ridge(alpha=10)
model = model.fit(X_train, y_train)

with open('Ridge_Model.sav', 'wb') as f:
    pickle.dump(model, f)

pre2 = model.predict(X_train)
print("train score Ridge : ")
print(model.score(X_train, y_train))
print("train mean squared error Ridge : ")
print(metrics.mean_squared_error(y_train, pre2))

dt = DecisionTreeRegressor(max_depth=4, max_features=15)
dt = dt.fit(X_train, y_train)

with open('DecisionTreeRegressor_Model.sav', 'wb') as f:
    pickle.dump(dt, f)

predictT = dt.predict(X_train)
print("train score Decision tree : ")
print(dt.score(X_train, y_train))
print("train mean squared error Decision tree : ")
print(metrics.mean_squared_error(y_train, predictT))

# plt.style.use('fivethirtyeight')
# LR
# for i in top_feature:
#    newxt = np.array(X_test[i])
#    fn.plot_best_fit(newxt.reshape(-1, 1), y_test, LR, i,'LinearRegression')

# for i in top_feature:
#    newxt = np.array(X_test[i])
#    fn.plot_best_fit(newxt.reshape(-1, 1), y_test, model, i,'RidgeRegression')


# ----------------------------------test---------------------------------------------------


print('number of unique values in X_test :')
print(X_test.nunique())

X_test = X_test.drop('URL', axis=1)
X_test = X_test.drop('Name', axis=1)
X_test = X_test.drop('ID', axis=1)
X_test = X_test.drop('Icon URL', axis=1)

print('number of null values in X_test :')
print(X_test.isna().sum())

X_test = X_test.drop('Description', axis=1)
X_test = X_test.drop('Primary Genre', axis=1)
X_test = X_test.drop('Subtitle', axis=1)

X_test = fn.in_app_Purchases_pros(X_test)

X_test = fn.AgeRate_pros(X_test)

X_test = fn.languages_pros(X_test)

df_encoded = fn.genres_features_pros(X_test)

X_test = X_test.drop('Genres', axis=1)

X_test['Original Release Date'] = pd.to_datetime(X_test['Original Release Date'], dayfirst=True)
X_test['timestamp OR'] = pd.DatetimeIndex(X_test['Original Release Date']).astype('int64')
X_test = X_test.drop('Original Release Date', axis=1)

X_test['Current Version Release Date'] = pd.to_datetime(X_test['Current Version Release Date'],
                                                        dayfirst=True)
X_test['timestamp CV'] = pd.DatetimeIndex(X_test['Current Version Release Date']).astype('int64')
X_test = X_test.drop('Current Version Release Date', axis=1)

'''
Q1_test = []
with open('Q1.pkl', 'rb') as f:
   Q1_test = pickle.load(f)

Q3_test = []
with open('Q3.pkl', 'rb') as f:
   Q3_test = pickle.load(f)

IQR_test = []
with open('IQR.pkl', 'rb') as f:
   IQR_test = pickle.load(f)
'''
"""
Q1_test = X_test.quantile(0.25)
Q3_test = X_test.quantile(0.75)

IQR_test = Q3_test - Q1_test
"""
'''  
Age_mode_test = 0
with open('Age_mode.pkl', 'rb') as f:
   Age_mode_test = pickle.load(f)

lang_mode_test = 0
with open('lang_mode.pkl', 'rb') as f:
   lang_mode_test = pickle.load(f)

user_mode_test = 0
with open('user_mode.pkl', 'rb') as f:
   user_mode_test = pickle.load(f)

tcv_mode_test = 0
with open('tcv_mode.pkl', 'rb') as f:
   tcv_mode_test = pickle.load(f)
'''
"""
Age = Age_mode  # int(Age_mode_test)
X_test['Age Rating'] = pd.Series(np.where(((X_test['Age Rating'] < (Q1_test[3] - 1.5 * IQR_test[3]))
                                           | (X_test['Age Rating'] > (Q3_test[3] + 1.5 * IQR_test[3]))) == True,
                                          Age, X_test['Age Rating']))

lang = lang_mode  # int(lang_mode_test)
X_test['Languages'] = pd.Series(np.where(((X_test['Languages'] < (Q1_test[4] - 1.5 * IQR_test[4]))
                                          | (X_test['Languages'] > (Q3_test[4] + 1.5 * IQR_test[4]))) == True,
                                         lang, X_test['Languages']))

user = user_mode  # int(user_mode_test)
X_test['User Rating Count'] = pd.Series(np.where(((X_test['User Rating Count'] < (Q1_test[0] - 1.5 * IQR_test[0]))
                                                  | (X_test['User Rating Count'] > (
                    Q3_test[0] + 1.5 * IQR_test[0]))) == True,
                                                 user, X_test['User Rating Count']))

tcv = tcv_mode  # int(tcv_mode_test)
X_test['timestamp CV'] = pd.Series(np.where(((X_test['timestamp CV'] < (Q1_test[7] - 1.5 * IQR_test[7]))
                                             | (X_test['timestamp CV'] > (Q3_test[7] + 1.5 * IQR_test[7]))) == True,
                                            tcv, X_test['timestamp CV']))
"""
X_test = fn.feature_scaling_test(X_test)

X_test = pd.concat([X_test, df_encoded], axis=1)

top_feature_test = []
with open('TopFeatures.pkl', 'rb') as f:
    top_feature_test = pickle.load(f)

X_test = X_test[top_feature_test]

pre1 = LR.predict(X_test)
print("test score LR : ")
print(LR.score(X_test, y_test))
print("test mean squared error LR : ")
print(metrics.mean_squared_error(y_test, pre1))

pre3 = model.predict(X_test)
print("test score Ridge : ")
print(model.score(X_test, y_test))
print("test mean squared error Ridge : ")
print(metrics.mean_squared_error(y_test, pre3))

yhat = dt.predict(X_test)
print("test score Decision tree : ")
print(dt.score(X_test, y_test))
print("test mean squared error Decision tree : ")
print(metrics.mean_squared_error(y_test, yhat))