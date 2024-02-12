import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from matplotlib import pyplot, pyplot as plt
from numpy import arange
from sklearn import preprocessing


def label_encoding(x, columns):
    for i in columns:
        lbl = LabelEncoder()
        lbl.fit(list(x[i].values))
        x[i] = lbl.transform(list(x[i].values))
    return x


def feature_scaling(x):
    index = x.index
    x = np.array(x)
    print("X_min")
    print(x.min(axis=0))
    with open('minFeatures.pkl', 'wb') as f:
        pickle.dump(x.min(axis=0), f)

    with open('maxFeatures.pkl', 'wb') as f:
        pickle.dump(x.max(axis=0), f)

    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - min(x[:, i])) / (max(x[:, i]) - min(x[:, i])))

    start_index = index[0]
    normalized_x = pd.DataFrame(normalized_x, columns=['User Rating Count', 'Price', 'In-app Purchases',
                                                       'Developer', 'Age Rating', 'Languages', 'Size',
                                                       'timestamp OR', 'timestamp CV'],
                                index=range(start_index, start_index + len(x)))
    return normalized_x


def feature_scaling_test(x):
    index = x.index
    x = np.array(x)
    minFeatures = []
    maxFeatures = []
    with open('minFeatures.pkl', 'rb') as f:
        minFeatures = pickle.load(f)
    with open('maxFeatures.pkl', 'rb') as f:
        maxFeatures = pickle.load(f)

    normalized_x = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
        normalized_x[:, i] = ((x[:, i] - minFeatures[i]) / (maxFeatures[i] - minFeatures[i]))

    start_index = index[0]
    normalized_x = pd.DataFrame(normalized_x, columns=['User Rating Count', 'Price', 'In-app Purchases',
                                                       'Developer', 'Age Rating', 'Languages', 'Size',
                                                       'timestamp OR', 'timestamp CV'],
                                index=range(start_index, start_index + len(x)))
    return normalized_x


def extract_features(x, name, alis):
    x[name] = pd.to_datetime(x[name], dayfirst=True)
    x["month" + alis] = x[name].dt.month
    x["day" + alis] = x[name].dt.day
    x["year" + alis] = x[name].dt.year


def pre_age_rating(age_rating):
    if age_rating >= 4 and age_rating < 9:
        return 0  # kids
    elif age_rating >= 9 and age_rating < 12:
        return 1  # child
    elif age_rating >= 12 and age_rating < 17:
        return 2  # teenager
    elif age_rating >= 17:
        return 3  # adult


def AgeRate_pros(X_train):
    X_train["Age Rating"] = X_train["Age Rating"].str.split("+", expand=True)[0]
    X_train["Age Rating"] = pd.to_numeric(X_train["Age Rating"], errors="coerce")
    X_train["Age Rating"] = X_train["Age Rating"].apply(pre_age_rating)
    return X_train


def in_app_Purchases_pros(X_train):
    Purchases = X_train['In-app Purchases'].fillna(0)
    index = X_train.index
    count = index[0]
    for x in Purchases:
        c = 0
        for w in list(str(x).split(",")):
            c = c + float(w)
        Purchases[count] = c
        count = count + 1
    X_train['In-app Purchases'] = Purchases
    return X_train


def languages_pros(X_train):
    X_train['Languages'] = X_train['Languages'].apply(lambda l: [] if pd.isnull(l) else l.split(','))

    X_train['Languages'] = [len(set(lst)) if lst is not None else 0 for lst in X_train['Languages']]

    return X_train


def genres_features_pros(X):
    genres_features = []
    with open('GenresFeatures.pkl', 'rb') as f:
        genres_features = pickle.load(f)
    X['Genres'] = X['Genres'].str.split(',')
    mlb = MultiLabelBinarizer()
    genre_mlb = mlb.fit_transform(X['Genres'])
    test_feature = list(mlb.classes_)

    index = X.index
    start_index = index[0]

    df_encoded = pd.DataFrame(genre_mlb, columns=test_feature,
                              index=range(start_index, start_index + len(X)))

    res_data = pd.DataFrame(0, columns=genres_features,
                            index=range(start_index, start_index + len(X)))

    for gen in genres_features:
        if gen in test_feature:
            res_data[gen] = df_encoded[gen]
        else:
            res_data[gen] = 0
    return res_data


'''
    for gen in genres_features:
        if gen in test_feature:
            continue
        else:
            df_encoded[gen] = 0

    for test_gen in test_feature:
        if test_gen not in genres_features:
            df_encoded = df_encoded.drop(test_gen, axis = 1)
'''


def Date_pros(X, col, newcol):
    X[col] = pd.to_datetime(X[col], dayfirst=True)
    X[newcol] = pd.DatetimeIndex(X[col]).astype(int)
    X = X.drop(col, axis=1)
    return X


def plot_best_fit(X, y, model, name, mod_name):
    # fut the model on all data
    model.fit(X, y)
    # plot the dataset
    pyplot.scatter(X, y)
    # plot the line of best fit
    xaxis = arange(X.min(), X.max(), 0.01)
    yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
    pyplot.xlabel(name, fontsize=20)
    pyplot.ylabel('Average User Rating', fontsize=20)
    pyplot.plot(xaxis, yaxis, color='r')
    # show the plot
    pyplot.title(mod_name)
    pyplot.show()


def bar_Graph_accuracy(model , accuracy):
    plt.bar(model, accuracy)
    plt.ylabel('Accuracy')
    plt.title("Model")
    plt.show()

def bar_graph_time(model , time ):
    plt.bar(model, time)
    plt.ylabel("Train time (seconds)")
    plt.title("Model")
    plt.show()

def bar_graph_time2(model , time ):
    plt.bar(model, time)
    plt.ylabel("Test time (seconds)")
    plt.title("Model")
    plt.show()


def convert_classes(data):
    for i in range(5214):
        if data.iloc[i]["Rate"] == 'Low':
            data.loc[i, "Rate"] = '0'
        elif data.iloc[i]["Rate"] == 'Intermediate':
            data.loc[i, "Rate"] = '1'
        elif data.iloc[i]["Rate"] == 'High':
            data.loc[i, "Rate"] = '2'
    return data[['Rate']]