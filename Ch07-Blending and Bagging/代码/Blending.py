import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model.stochastic_gradient import SGDClassifier

# 边际可视化
def plot_decision_regions(X, y, ax, classifier=None, resolution=0.02, need_samples=1):
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	if classifier:
		x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
		x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
		xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
								np.arange(x2_min,x2_max,resolution))
		temp = np.array([xx1.ravel(),xx2.ravel()]).T
		Z = classifier.predict(temp)
		Z = Z.reshape(xx1.shape)
		ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
		ax.set_xlim(xx1.min(), xx1.max())
		ax.set_ylim(xx2.min(), xx2.max())
	if need_samples:
		for idx,cl in enumerate(np.unique(y)):
			ax.scatter(x=X[y==cl,0],
						y=X[y==cl,1],
						alpha=0.8,
						c=colors[idx],
						marker=markers[idx],
						label=cl,
						edgecolor='black')

# 绘出数据点
def plot_samples(X,y):
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','gray','cyan')
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0],
		            y=X[y==cl,1],
		            alpha=0.8,
		            c=colors[idx],
		            marker=markers[idx],
		            label=cl,
		            edgecolor='black')
		plt.legend()
	plt.show()

# Train-Val-Test分拆
def train_val_test_split(X, y, val_size=0.2, test_size=0.2):
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1)
    X_train,X_val,y_train,y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train, random_state=1)
    return X_train, X_val, X_test, y_train, y_val, y_test

# 线性混合模型
class LB_model:
	def __init__(self, estimators):
		self.estimators = estimators

	def get_X_transformed(self, X):
		X_transformed = np.vstack([i.predict(X) for i in self.estimators.values()]).T
		return X_transformed

	def fit(self, X_val, y_val, X_train, y_train):
		# 保存类别标签
		self.classes_ = np.unique(y_train)
		X_transformed = self.get_X_transformed(X_val)
		sgd = SGDClassifier(max_iter=2000, random_state=1)
		sgd.fit(X_transformed, y_val)
		self.blender_ = sgd
		print("alpha拟合完毕, 现重新训练g")
		X = np.vstack([X_train,X_val])
		y = np.hstack([y_train,y_val])
		for i in self.estimators.values():
			i.fit(X,y)
		print("g重新训练完毕, 线性混合模型训练完毕")
            
	def predict(self, X):
		X_transformed = self.get_X_transformed(X)
		result = self.blender_.predict(X_transformed)
		threshold = self.classes_.mean()
		result = np.where(result>=threshold, self.classes_.max(), self.classes_.min())
		return result

	def score(self, X, y):
		result = self.predict(X)
		accuracy = (result == y).sum()/len(y)
		return accuracy

if __name__ == '__main__':
	# Step 1: Create some data
	print("*********Step 1*********")
	print("生成学习集合，并作图")
	X,y = make_moons(n_samples=1000, noise=0.6, random_state=1)
	plot_samples(X,y)

	# Step 2: 拆分
	print("*********Step 2*********")
	print("Train-Val-Test分拆")
	X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X,y)
	print("训练集%s笔数据，验证集%s笔数据，测试集%s笔数据" % (str(len(y_train)),str(len(y_val)),str(len(y_test))))

	# Step 3: 训练不同的g
	print("*********Step 3*********")
	print("训练不同的g")
	lr = LogisticRegression(C=0.1, random_state=1)
	lr.fit(X_train, y_train)	
	tree = DecisionTreeClassifier(min_samples_leaf = 4, random_state=1)
	tree.fit(X_train, y_train)
	svm = SVC(kernel='rbf', gamma=1, C=1, random_state=1)
	svm.fit(X_train, y_train)

	# Step 4: 线性混合-Linear Blending
	print("*********Step 4*********")
	print("开始线性混合")
	estimators = {'LogisticRegression':lr,'DecisionTree':tree, 'SVM':svm}
	lb = LB_model(estimators)
	lb.fit(X_val, y_val, X_train, y_train)

	# Step 5: 线性混合模型的表现
	print("*********Step 5*********")
	print("测试线性混合模型的表现")
	lb_score = lb.score(X_test, y_test)
	print("线性混合模型的准确率: %s" % str(lb_score))

	# Step 6: 单一模型的表现
	print("*********Step 6*********")
	print("单一模型的表现")
	lr_score = lr.score(X_test, y_test)
	print("LogisticRegression表现: %s" % str(lr_score))
	tree_score = tree.score(X_test, y_test)
	print("DecisionTree表现: %s" % str(tree_score))
	svm_score = svm.score(X_test, y_test)
	print("SVM表现: %s" % str(svm_score))

	# Step 7: 各模型边界作图
	print("*********Step 7*********")
	print("各模型边界作图: 基于Test Set")
	plt.figure(figsize=(20,3))
	gs = gridspec.GridSpec(1,4)
	ax1 = plt.subplot(gs[0,0])
	ax2 = plt.subplot(gs[0,1])
	ax3 = plt.subplot(gs[0,2])
	ax4 = plt.subplot(gs[0,3])
	ax1.set_title("LogisticRegression")
	plot_decision_regions(X_test, y_test, ax=ax1, classifier=lr)
	ax2.set_title("DecisionTree")
	plot_decision_regions(X_test, y_test, ax=ax2, classifier=tree)
	ax3.set_title("SVM")
	plot_decision_regions(X_test, y_test, ax=ax3, classifier=svm)
	ax4.set_title("Linear-Blending")
	plot_decision_regions(X_test, y_test, ax=ax4, classifier=lb)
	plt.show()


# 线性混合模型的准确率:		0.795
# LogisticRegression表现:	0.79
# DecisionTree表现:			0.695
# SVM表现: 					0.79