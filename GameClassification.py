import time
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier, \
    GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB

import functions as fn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("games-classification-dataset.csv")

X = data.iloc[:, 0:17]

Y = fn.convert_classes(data)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,
                                                    shuffle=False)


colname = ['URL', 'Name', 'Description', 'Developer', 'Primary Genre', 'Subtitle']
for i in colname:
    lbl = LabelEncoder()
    lbl.fit(list(X_train[i].values))
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
X_train['timestamp OR'] = pd.DatetimeIndex(X_train['Original Release Date']).astype(int)
X_train = X_train.drop('Original Release Date', axis=1)

X_train['Current Version Release Date'] = pd.to_datetime(X_train['Current Version Release Date'],
                                                         dayfirst=True)
X_train['timestamp CV'] = pd.DatetimeIndex(X_train['Current Version Release Date']).astype(int)
X_train = X_train.drop('Current Version Release Date', axis=1)

X_train = fn.feature_scaling(X_train)

X_train = pd.concat([X_train, df_encoded], axis=1)

fvalue_Best = SelectKBest(f_classif, k=15)
X_kbest = fvalue_Best.fit_transform(X_train, y_train)

feature_names = X_train.columns[fvalue_Best.get_support()]

with open('TopFeatures_classification.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# -----------Classifications ---------------

# ---------- Gradient boosting classifier -------------
"""
param_grid = {
    'n_estimators': [100],
    'max_depth': [3],
    'learning_rate': [0.1]
}
gb_clf = GradientBoostingClassifier()
grid_search = GridSearchCV(gb_clf, param_grid, cv=5)
start_time=time.time()
Gradient_Boosting=grid_search.fit(X_kbest, y_train)
end_time = time.time()
train_time1 = end_time - start_time
# save model
with open('Gradient_Boosting.pkl', 'wb') as f:
    pickle.dump(Gradient_Boosting, f)

print("Best hyper-parameters :", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
y_pred_gnb = grid_search.predict(X_kbest)
accuracy = accuracy_score(y_train, y_pred_gnb)
print("Train Accuracy for Gradient Boosting :", accuracy)
"""
# ---------------logistic regression , GaussianNB , with stacking Classifier ----------
"""
lr_ovr = OneVsRestClassifier(LogisticRegression(penalty='l2',C=0.1,solver='saga',max_iter=5)).fit(X_kbest, y_train)
nb = GaussianNB()
rf = RandomForestClassifier()
# Create a stacking classifier with logistic regression, naive Bayes, and random forest as the base estimators
stacking = StackingClassifier(estimators=[('lr', lr_ovr), ('nb', nb), ('rf', rf)])
start_time=time.time()
stack_classifier=stacking.fit(X_kbest, y_train)
end_time=time.time()
train_time2=end_time-start_time
# save model
with open('stack_classifier2.pkl', 'wb') as f:
    pickle.dump(stack_classifier, f)
"""
# c = 1 --> 0.566
# c = 100 --> 0.564
# c = 0.1 --> 0.57188
# c = 0.1 , solver = liblinear , accuracy = 0.57316
# c = 0.1 , solver saga , accuracy 0.5750798722044729  max iter = 5
# -----------------SVM-------------------------
"""
C = 10000
rf = RandomForestClassifier(n_estimators=10, max_depth=None)
start_time=time.time()
rbf_svc = svm.SVC(kernel='rbf', gamma=50, C=C,probability=True).fit(X_kbest, y_train)
rf.fit(X_kbest, y_train)
end_time=time.time()
train_time_estimator=end_time-start_time

voting_clf = VotingClassifier(
    estimators=[('svm', rbf_svc), ('rf', rf)],
    voting='soft'
)
start_time=time.time()
vote_classifier=voting_clf.fit(X_kbest, y_train)
end_time=time.time()
train_time=end_time - start_time
# save model
with open('vote_classifier.pkl', 'wb') as f:
    pickle.dump(vote_classifier, f)
train_time_all = train_time_estimator+train_time
#time2=[train_time1,train_time2,train_time_all]
#fn.bar_graph_time(["grid_search","stacking","voting_clf"],time2)
"""
# c =0.00000000000000000000001  , gamma=0.8  , accuracy = 0.49328583173472185 , kfeatures = 24
# c = 0.1 , gamma = 0.1,accuracy = 0.5609756097560976 ,  kfeatures = 24
# c = 0.1 , gamma = 5 , accuracy = 0.5878322828172102 ,  kfeatures = 24
# c = 0.1 , gamma = 5 , accuracy = 0.5919429980816662 ,  kfeatures = 20
# c = 5   , gamma = 5 , accuracy = 0.5963277610304193 ,  kfeatures = 10
# c = 10000 , gamma = 5 ,  accuracy = 0.6692244450534393 , kfeatures = 10
# c = 10000 , gamma = 10 , accuracy = 0.7089613592765142 , kfeatures = 10
# c = 10000 , gamma = 50 , accuracy = 0.8651685393258427 , kfeatures = 10 ,  ---- test accuracy = 0.5201277955271566
# c = 5000  , gamma = 20 , accuracy = 0.825979720471362  , kfeatures = 15 ,  ---- test accuracy = 0.5488817891373802
# c = 2000  , gamma = 20 , accuracy = 0.8109070978350233 , kfeatures = 15 ,  ---- test accuracy = 0.5559105431309904
# c = 1000  , gamma = 20 , accuracy = 0.793642093724308  , kfeatures = 15 ,  ---- test accuracy = 0.5654952076677316
# c = 0.1   , gamma = 20 , accuracy = 0.5987941901890929 , kfeatures = 15 ,  ---- test accuracy = 0.5814696485623003

# -------------------test----------------------------

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
X_test['timestamp OR'] = pd.DatetimeIndex(X_test['Original Release Date']).astype(int)
X_test = X_test.drop('Original Release Date', axis=1)

X_test['Current Version Release Date'] = pd.to_datetime(X_test['Current Version Release Date'],
                                                        dayfirst=True)
X_test['timestamp CV'] = pd.DatetimeIndex(X_test['Current Version Release Date']).astype(int)
X_test = X_test.drop('Current Version Release Date', axis=1)

X_test = fn.feature_scaling_test(X_test)

X_test = pd.concat([X_test, df_encoded], axis=1)

# ----------------Classifications for test data -----------------
top_feature_test = []
with open('TopFeatures_classification.pkl', 'rb') as f:
    top_feature_test = pickle.load(f)

X_test = X_test[top_feature_test]
# ------------ Gradient Boosting -------------

with open('Gradient_Boosting.pkl', 'rb') as f:
    GB = pickle.load(f)
start_time=time.time()
test_pred =  GB.predict(X_test)
end_time=time.time()
test_time1=end_time-start_time
test_accuracy1 = accuracy_score(y_test, test_pred)
print("test Accuracy of gird search :", test_accuracy1)

# ----------------- logistic Regression Only--------------
"""
accuracy = lr_ovr.score(X_test, y_test)
print('OneVsRest Logistic Regression accuracy Test : ' + str(accuracy))
"""
# ----------------- logistic , random forest with stacking -----------

with open('stack_classifier2.pkl', 'rb') as f:
    SC = pickle.load(f)
start_time=time.time()
y_pred = SC.predict(X_test)
end_time=time.time()
test_time2=end_time-start_time
accuracy2 = accuracy_score(y_test, y_pred)
print("Accuracy after stacking Test : "+ str(accuracy2))

# accuracy after bagging : 0.5731629392971246
# accuracy after stacking with gaussian NB and random forest = 0.6057507987220447
# accuracy after random forest ==  0.594888178913738
# random forest and stacking classifier 0.612779552715655

# ---------------- SVM with kernal RBF
"""
predictions_test = rbf_svc.predict(X_test)
accur_test = accuracy_score(predictions_test, y_test)
print("Accuracy of SVM in Test : "+str(accur_test))
"""
# --------- using boosting to optimize the classifier Random forest with voting classifier ------------------

with open('vote_classifier.pkl', 'rb') as f:
    VC = pickle.load(f)
start_time=time.time()
y_pred=VC.predict(X_test)
end_time=time.time()
test_time3=end_time-start_time
time_test=[test_time1,test_time2,test_time3]
fn.bar_graph_time2(["grid_search","stacking","voting_clf"],time_test)
ensemble_accuracy = accuracy_score(y_test,y_pred)
print("Test accuracy with RBF SVM After Ensemble : ", ensemble_accuracy)
acc=[test_accuracy1,accuracy2,ensemble_accuracy]
fn.bar_Graph_accuracy(["grid_search","stacking","voting_clf"],acc)

# stack classifier
# Accuracy after stacking Test : 0.6212847555129435 "stack_classifier2.pkl"
# Accuracy after stacking Test : 0.6164908916586769 "stack_classifier.pkl"

# svm with random forest , voting classifier
# Accuracy of SVM train : 0.9095642641819677
# Accuracy of SVM in Test : 0.5143769968051118
# Test accuracy with RBF SVM After Ensemble :  0.5896452540747843

# gradient Boosting
# Best score: 0.6142397219948592
# Train Accuracy for Gradient Boosting : 0.6964756653080796
# test Accuracy: 0.5963566634707574


