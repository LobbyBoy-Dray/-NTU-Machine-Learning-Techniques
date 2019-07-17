import numpy as np
import pandas as pd

class RidgeRegressor:
    def __init__(self, lamb=0.1):
        self.lamb = lamb

    def addX0(self, X):
        if len(X.shape)<2:
            X = np.hstack([1,X])
        else:
            N = X.shape[0]
            X = np.hstack([np.ones(N).reshape(N,1),X])
        return X

    def fit(self, X, y):
        X = self.addX0(X)
        I = np.eye(X.shape[1])
        self.w_ = np.linalg.inv(self.lamb*I + X.T.dot(X)).dot(X.T).dot(y)
        return self

    def predict(self, X):
        X = self.addX0(X)
        y_predict = np.sign(X.dot(self.w_))
        y_predict = np.where(y_predict==0, +1, y_predict)
        return y_predict
    
    def error(self, X, y):
        N = len(y)
        y_predict = self.predict(X)
        error01   = (y!=y_predict).sum()/N
        return error01

class RidgeRegressionBagging:
    def __init__(self, lamb, T, boot_sample, random_state=1126):
        self.lamb = lamb
        self.T = T
        self.boot_sample = boot_sample
        self.random_state = random_state

    def fit(self, X, y):
        self.estimators_ = []
        rng = np.random.RandomState(self.random_state)
        for t in range(self.T):
            # resampling
            index = rng.choice(range(X.shape[0]), self.boot_sample, replace=True)
            X_t   = X[index]
            y_t   = y[index]
            # training
            estimator = RidgeRegressor(lamb=self.lamb)
            estimator.fit(X_t, y_t)
            self.estimators_.append(estimator)
        return self

    def predict(self, X):
        result = np.array([i.predict(X) for i in self.estimators_])
        result = np.sign(result.sum(axis=0))
        result = np.where(result==0, +1, result)
        return result

    def error(self, X, y):
        N = len(y)
        y_predict = self.predict(X)
        error01   = (y!=y_predict).sum()/N
        return error01

def p9():
    data = np.loadtxt("./data/hw2_lssvm_all.dat")
    X_train = data[:400,:-1]
    y_train = data[:400,-1]
    X_test  = data[400:,:-1]
    y_test  = data[400:,-1]
    lambdas = [0.05, 0.5, 5, 50, 500]
    for lamb in lambdas:
        print("****lambda: %s****" % lamb)
        ridge_reg = RidgeRegressor(lamb=lamb)
        ridge_reg.fit(X_train,y_train)
        E_in = ridge_reg.error(X_train, y_train)
        print("E_in: %s" % E_in)
        E_out = ridge_reg.error(X_test, y_test)
        print("E_out: %s" % E_out)

def p10():
    data = np.loadtxt("./data/hw2_lssvm_all.dat")
    X_train = data[:400,:-1]
    y_train = data[:400,-1]
    X_test  = data[400:,:-1]
    y_test  = data[400:,-1]
    lambdas = [0.05, 0.5, 5, 50, 500]
    for lamb in lambdas:
        print("****lambda: %s****" % lamb)
        bagging = RidgeRegressionBagging(lamb=lamb, T=250, boot_sample=400)
        bagging.fit(X_train,y_train)
        E_in = bagging.error(X_train, y_train)
        print("E_in: %s" % E_in)
        E_out = bagging.error(X_test, y_test)
        print("E_out: %s" % E_out)

if __name__ == "__main__":
    print("========= 9 =========")
    p9()
    print("========= 10 =========")
    p10()
