import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 边际可视化
def plot_bound(X, y, classifier, color='black', alpha=1, margin=0.1, linewidths=2, resolution=0.005):
	x1_min, x1_max = X[:,0].min()-margin, X[:,0].max()+margin
	x2_min, x2_max = X[:,1].min()-margin, X[:,1].max()+margin
	xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
							np.arange(x2_min,x2_max,resolution))
	temp = np.array([xx1.ravel(),xx2.ravel()]).T
	Z = classifier.predict(temp)
	Z = Z.reshape(xx1.shape)
	C = plt.contour(xx1, xx2, Z, 0, colors=color, alpha=alpha, linewidths=linewidths)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

# 绘出数据点
def plot_samples(X,y):
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','gray','cyan')
	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl,0],
		            y=X[y==cl,1],
		            alpha=0.8,
		            s = 120,
		            c=colors[idx],
		            marker=markers[idx],
		            label=cl,
		            edgecolor='black')
		plt.legend()

# Bootstrap
def Bootstrapping(X,y,rgen = None):
	m = len(y)
	if rgen:
		new_index = rgen.choice(range(m),m,replace=True)
	else:
		new_index = np.random.choice(range(m),m,replace=True)
	return X[new_index],y[new_index]

class PocketClassifier:
	# updates: 更新口袋的次数;
	# 此外, 并没有: 每次随机抽取一个样本, 看其是否错误, 错误则改正, 没错则继续抽取;
	# 因为这样对于错误极少的有偏数据集来说, 训练太慢, 每次可能抽的都是没错的;
	# 本算法每次从犯错的点中, 随机抽取一个, 保证每次都能改正, 提高速度。
	def __init__(self, updates=1000, random_state=1):
		self.updates = updates
		self.rgen = np.random.RandomState(random_state)

	def fit(self, X_train, y_train):
		m = X_train.shape[0]
		n = X_train.shape[1]
		self.w_ = np.zeros(n+1)
		self.w_best_ = np.zeros(n+1)
		self.w_best_errors_, error_index = self.errors(self.predict(X_train, use_best=1), y_train)
		if self.w_best_errors_ == 0:
			print("炒鸡好的运气, 不需迭代直接结束")
			return self
		now_iter = 0
		while now_iter < self.updates:
			now_iter += 1
			self.w_[1:] += y_train[error_index] * X_train[error_index]
			self.w_[0] += y_train[error_index]
			new_errors, error_index = self.errors(self.predict(X_train, use_best=0), y_train)
			if new_errors < self.w_best_errors_:
				self.w_best_ = self.w_.copy()
				self.w_best_errors_ = new_errors
			if new_errors == 0:
				print("100%训练集正确率, 停止迭代, 结束")
				return self
		return self

	def error_rate(self, X_test, y_test):
		y_hat = self.predict(X_test)
		errors = self.errors(y_hat, y_test)
		return errors/X_test.shape[0]

	# 返回错误数量; 如果错误数量不为0, 同时返回任意一个错误的索引
	def errors(self, y_hat, y):
		temp = (y_hat != y)
		error_num = temp.sum()
		error_index = -1 if error_num==0 else self.rgen.choice(np.where(temp==True)[0],1)[0]
		return error_num, error_index

	def predict(self, X, use_best=1):
		w_ = self.w_best_ if use_best else self.w_
		net_input = np.dot(X,w_[1:])+w_[0]
		output = np.sign(net_input)
		output = np.where(output==0, -1, output)
		return output

class Bagging_PocketClassifier:
	def __init__(self, estimators):
		self.estimators = estimators

	def predict(self, X):
		result = 0
		for ppn in self.estimators:
			result += ppn.predict(X)
		result = np.sign(result)
		result = np.where(result==0, -1, result)
		return result

if __name__ == '__main__':
	# Step 1: Create some data
	print("*********Step 1*********")
	print("设定数据集")
	X = np.array([[0.6,1.5,2.1,2.3,3.3,3.5,4.7,-0.2,-0.5,-1.5,-1.7,1.1,1.6,-0.2,-0.2,-0.1,-0.05,-0.2,-0.25,],
		          [1.2,4.7,2.1,0.9,1.6,2,1.3,4.9,4.5,0.7,0.9,-2,1-0.7,-0.2,-1.5,-3.5,-4.1,-4,-4.2]]).T
	y = np.array([1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
	plot_samples(X,y)
	plt.show()

	# Step 2: 训练多个PocketClassifier
	print("*********Step 3*********")
	n_classifiers = 25
	classifiers = []
	print("训练%s个PocketClassifier-Bootstarpping法" % str(n_classifiers))
	updates = 1000
	for i in range(n_classifiers):
		rgen = np.random.RandomState(i+100)
		X_train_temp, y_train_temp = Bootstrapping(X, y, rgen=rgen)
		ppn = PocketClassifier(updates=updates, random_state=i)
		ppn.fit(X_train_temp, y_train_temp)
		classifiers.append(ppn)

	# Step 3: Uniform Blending
	print("*********Step 4*********")
	print("Uniform聚合")
	BP = Bagging_PocketClassifier(classifiers)

	# Step 4: 在同一张图上绘出数据集+所有Pocket分类器+Bagging分类器
	print("*********Step 5*********")	
	print("在同一张图上绘出数据集+所有Pocket分类器+Bagging分类器")
	plt.figure()
	plot_samples(X, y)
	for ppn in classifiers:
		plot_bound(X, y, classifier=ppn, color='gray', alpha=0.2)
	plot_bound(X, y, classifier=BP, linewidths=4)
	plt.xticks(())
	plt.yticks(())
	plt.title("$T_{POCKET}=%s;\ T_{BAG}=%s$" % (str(updates), str(n_classifiers)))
	plt.show()




