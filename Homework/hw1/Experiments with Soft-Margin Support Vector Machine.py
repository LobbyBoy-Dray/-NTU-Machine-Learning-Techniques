import requests
import re
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import matplotlib.pyplot as plt

def download_data():
    url_train = "http://www.amlbook.com/data/zip/features.train"
    url_test  = "http://www.amlbook.com/data/zip/features.test"
    r_train   = requests.get(url_train)
    r_test    = requests.get(url_test)
    train     = r_train.text
    test      = r_test.text
    df_train  = pd.DataFrame([re.split(' +', i)[1:] for i in train.split('\n')[:-1]])
    df_test   = pd.DataFrame([re.split(' +', i)[1:] for i in test.split('\n')[:-1]])
    df_train.to_csv("trian.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)

############ Experiments with Soft-Margin Support Vector Machine ############
df_train = pd.read_csv("trian.csv", header=None)
df_test  = pd.read_csv("test.csv", header=None)
##### 13 #####
def p13():
    X_train = df_train.iloc[:, [1,2]].values
    y_train = df_train.iloc[:, 0].values
    y_train = np.where(y_train==2, +1, -1)
    Clog10  = [-5,-3,-1,+1,+3] 
    for c in Clog10:
        svc = SVC(kernel='linear', C=10**c)
        svc.fit(X_train,y_train)
        print("log10-C: %s" % (c))
        print(np.sqrt(svc.coef_.dot(svc.coef_.T))[0][0])

##### 14 #####
def p14():
    X_train = df_train.iloc[:, [1,2]].values
    y_train = df_train.iloc[:, 0].values
    y_train = np.where(y_train==4, +1, -1)
    Clog10  = [-5,-3,-1,+1,+3] 
    for c in Clog10:    
        svc = SVC(kernel='poly', degree=2, gamma=1, coef0=1, C=10**c)
        svc.fit(X_train,y_train)
        print("log10-C: %s" % (c))
        print("Accuracy: %s" % svc.score(X_train,y_train))

##### 15 #####
def p15():
    X_train = df_train.iloc[:, [1,2]].values
    y_train = df_train.iloc[:, 0].values
    y_train = np.where(y_train==0, +1, -1)
    Clog10  = [-2,-1,0,+1,+2] 
    for c in Clog10:
        print("log10-C: %s" % (c))
        svc = SVC(kernel='rbf', gamma=80, C=10**c)
        svc.fit(X_train,y_train)
        margin_square = 1/(svc.dual_coef_).dot(svc.support_vectors_.dot(svc.support_vectors_.T)).dot(svc.dual_coef_.T)[0][0]
        margin        = np.sqrt(margin_square)
        print(margin)

##### 16 #####
# exited in 1575.301 seconds
def p16():
    X_train = df_train.iloc[:, [1,2]].values
    y_train = df_train.iloc[:, 0].values
    y_train = np.where(y_train==0, +1, -1)
    N = len(y_train)
    count = {-2:0, -1:0, 0:0, 1:0, 2:0}
    for t in range(100):
        # 构造含有1000个样本的验证集
        test_fold = -np.ones(N)
        val_index = np.random.choice(range(len(y_train)), 1000, replace=False)
        test_fold[val_index] = 0
        val_set = PredefinedSplit(test_fold)
        svc = SVC(kernel='rbf', C=0.1)
        gs = GridSearchCV(estimator=svc, param_grid={'gamma':np.float_power(10,[-2,-1,0,1,2])}, cv=val_set)
        gs.fit(X_train, y_train)
        count[np.log10(gs.best_params_['gamma'])] += 1

    plt.bar(count.keys(), count.values(), color='red')
    plt.xlabel("$log_{10}\gamma$")
    plt.ylabel("Selected Times")
    plt.show()

