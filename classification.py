import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.utils import shuffle
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

def get_train_test():
    # df = pd.read_csv('data_processed_25/data_all_feature_entity_semantic.csv')[['target', 'entity', 'backchannel', 'semantic', 'lda', 'liwc', 'prosociety']]
    # df = shuffle(df).reset_index(drop=True)
    # df.to_csv('best/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.csv')
    # print(df)
    df = pd.read_csv('best/2022-06-06-04-41-22.csv')
    target_all = df['target']
    feature_all = []
    print(target_all)
    for index in range(len(df['entity'])):
        e = literal_eval(df['entity'][index])
        b = [df['backchannel'][index]]
        s = [df['semantic'][index]]
        l = literal_eval(df['lda'][index])
        li = literal_eval(df['liwc'][index])
        p = literal_eval(df['prosociety'][index])
        print(len(e))
        feature_all.append(e+b+li)

    return feature_all, target_all

def classify():
    feature_all, target_all= get_train_test()
    cls = RandomForestClassifier(n_estimators=500)
    # cls = SGDClassifier(loss='log')
    # cls = SVC(kernel='rbf')
    # cls = GaussianNB()
    # cls = DecisionTreeClassifier()
    score =  cross_val_score(cls, feature_all, target_all, cv=5,scoring='accuracy')
    print(score.mean())
    print(score)
    feature_train = feature_all[:int(0.7*(len(feature_all)))]
    target_train = target_all[:int(0.7*(len(feature_all)))]
    feature_test = feature_all[int(0.7*(len(feature_all))):]
    target_test = target_all[int(0.7*(len(feature_all))):]
    cls.fit(feature_train, target_train)
    result =cls.predict(feature_test)
    print(accuracy_score(target_test, result))
    try:
        print(cls.feature_importances_)
    except:
        pass






if __name__ == "__main__":
    classify()