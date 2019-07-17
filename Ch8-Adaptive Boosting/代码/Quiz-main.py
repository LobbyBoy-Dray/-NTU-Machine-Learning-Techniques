from DecisionStump import DecisionStumpClassifier
from Plotter import plot_data_or_bound
from AdaBoost import AdaBooster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def dat_to_csv(filename):
	with open(filename, encoding='utf-8') as f:
		lines = f.readlines()
	csv_text = '\n'.join([','.join(i.strip().split(' ')) for i in lines])
	filename = filename.replace('.dat', '.csv')
	with open(filename, 'w', encoding='utf-8') as f:
		f.write(csv_text)
	return filename

if __name__ == '__main__':
	# Step 1: read data from .dat file
	train_path = "./data/hw2_adaboost_train.dat"
	test_path  = "./data/hw2_adaboost_test.dat"
	train_path = dat_to_csv(train_path)
	test_path  = dat_to_csv(test_path)
	Train = pd.read_csv(train_path, header=None).values
	Test  = pd.read_csv(test_path, header=None).values
	X_train, y_train = Train[:,:-1], Train[:,-1]
	X_test, y_test   = Test[:,:-1], Test[:,-1]
	# Step 2: Train AdaBoost model
	T = 500
	print("*********T=%s*********" % str(T))
	adabooster = AdaBooster(base=DecisionStumpClassifier, T=T)
	adabooster.fit(X_train,y_train)
	plot_data_or_bound(X_train,y_train,classifier=adabooster, resolution=0.02)
	plt.show()
	# Step 3: every g_t E_in
	g_classifiers = adabooster.g_classifier_
	g_t_E_in        = []
	for g in g_classifiers:
		single_E_in = ((g.predict(X_train) != y_train).sum())/len(y_train)
		g_t_E_in.append(single_E_in)
	plt.figure(figsize=(15,4))
	plt.title("$E_{in}(g_t)$", fontsize=18)
	plt.plot(range(T),g_t_E_in,color='red')
	plt.show()
	# Step 4: every G_t E_in
	g_alpha, g_classifiers = adabooster.g_alpha_, adabooster.g_classifier_
	G_t_E_in = []
	for i in range(T):
		voting_basket = 0
		for g,alpha in zip(g_classifiers[:i+1], g_alpha[:i+1]):
			voting_basket += alpha * (g.predict(X_train))
		result = np.sign(voting_basket)
		result = np.where(result == 0, -1, result)
		single_E_in = ((result != y_train).sum())/len(y_train)
		G_t_E_in.append(single_E_in)
	plt.figure(figsize=(15,4))
	plt.title("$E_{in}(G_t)$", fontsize=18)
	plt.plot(range(T),G_t_E_in,color='red')
	plt.show()
	# Step 5: every U_t
	U = adabooster.U_
	plt.figure(figsize=(15,4))
	plt.title("$U_t$", fontsize=18)
	plt.plot(range(T+1),U,color='red')
	plt.show()
	# Step 6: every U_t
	g_alpha, g_classifiers = adabooster.g_alpha_, adabooster.g_classifier_
	G_t_E_out = []
	for i in range(T):
		voting_basket = 0
		for g,alpha in zip(g_classifiers[:i+1], g_alpha[:i+1]):
			voting_basket += alpha * (g.predict(X_test))
		result = np.sign(voting_basket)
		result = np.where(result == 0, -1, result)
		single_E_out = ((result != y_test).sum())/len(y_test)
		G_t_E_out.append(single_E_out)
	plt.figure(figsize=(15,4))
	plt.title("$E_{out}(G_t)$", fontsize=18)
	plt.plot(range(T),G_t_E_out,color='red')
	plt.show()


