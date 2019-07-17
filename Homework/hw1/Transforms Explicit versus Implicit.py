import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import solvers, matrix
from sklearn.svm import SVC

############ Transforms: Explicit versus Implicit ############
# Raw data
X = np.array([[1,0],
              [0,1],
              [0,-1],
              [-1,0],
              [0,2],
              [0,-2],
              [-2,0]])
y = np.array([-1,-1,-1,1,1,1,1])
##### 1 #####
def p1():
    # Feature Transform
    phi_1 = 2*(X[:,1]**2)-4*X[:,0]+2
    phi_2 = X[:,0]**2-2*X[:,1]-3
    Z     = np.array([phi_1,phi_2]).T
    # Draw the scatter picture
    plt.scatter(Z[y==-1,0], Z[y==-1,1], marker='o', color='blue')
    plt.scatter(Z[y==1,0], Z[y==1,1], marker='x', color='red')
    plt.show()
    # Accurate solution
    svc = SVC(kernel='linear', C=np.infty)
    svc.fit(Z, y)
    print("b: %s" % svc.intercept_)
    print("w: %s" % svc.coef_ )

##### 2 #####
def p2():
    svc = SVC(kernel='poly', C=np.infty, coef0=1, gamma=1, degree=2)
    svc.fit(X, y)
    print(svc.support_)
    print(svc.dual_coef_/y[svc.support_])

##### 3 #####
def p3():
    svc = SVC(kernel='poly', C=np.infty, coef0=1, gamma=1, degree=2)
    svc.fit(X, y)
    coef = np.zeros(6)
    for y_alpha,sv in zip(svc.dual_coef_[0], svc.support_vectors_):
        coef += np.array([1,2,2,1,1,2]) * np.array([1,sv[0],sv[1],sv[0]*sv[0],sv[1]*sv[1],2*sv[0]*sv[1]]) * y_alpha
    print(coef)
    print(svc.intercept_)


p3()