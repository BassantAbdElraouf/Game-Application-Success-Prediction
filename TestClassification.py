import pandas as pd
from sklearn.metrics import accuracy_score
import functions as fn
import numpy as np
import pickle
import time

data_of_test = pd.read_csv("D:/6th Term/Pattern/Project/GameProject/games-classification-dataset.csv")
X_of_test = data_of_test.iloc[:, 0:17]
Y_of_test=fn.convert_classes(data_of_test)
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
with open('TopFeatures_classification.pkl', 'rb') as f:
    top_feature_test = pickle.load(f)

X_of_test = X_of_test[top_feature_test]

with open('Gradient_Boosting.pkl', 'rb') as f:
    GB = pickle.load(f)
start_time=time.time()
test_pred =  GB.predict(X_of_test)
end_time=time.time()
test_time1=end_time-start_time
test_accuracy1 = accuracy_score(Y_of_test, test_pred)
print("Test Accuracy of Gradient Boosting Classifier :", test_accuracy1)

with open('stack_classifier2.pkl', 'rb') as f:
    SC = pickle.load(f)
start_time=time.time()
y_pred = SC.predict(X_of_test)
end_time=time.time()
test_time2=end_time-start_time
accuracy2 = accuracy_score(Y_of_test, y_pred)
print("Test Accuracy of stacking Classifier : "+ str(accuracy2))


with open('vote_classifier.pkl', 'rb') as f:
    VC = pickle.load(f)
start_time=time.time()
y_pred=VC.predict(X_of_test)
end_time=time.time()
test_time3=end_time-start_time
time_test=[test_time1,test_time2,test_time3]
fn.bar_graph_time2(["grid_search","stacking","voting_clf"],time_test)
ensemble_accuracy = accuracy_score(Y_of_test,y_pred)
acc=[test_accuracy1,accuracy2,ensemble_accuracy]
fn.bar_Graph_accuracy(["grid_search","stacking","voting_clf"],acc)
print("Test accuracy with RBF SVM After Ensemble with using Random forest : ", ensemble_accuracy)
