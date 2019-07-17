import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_data_or_bound(X, y, classifier=None, color='black', alpha=1, margin=0.1, linewidths=2, resolution=0.005):
	markers = ('s','x','o','^','v')
	colors = ('red','blue','lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	if classifier:
		x1_min, x1_max = X[:,0].min()-margin, X[:,0].max()+margin
		x2_min, x2_max = X[:,1].min()-margin, X[:,1].max()+margin
		xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
								np.arange(x2_min,x2_max,resolution))
		temp = np.array([xx1.ravel(),xx2.ravel()]).T
		Z = classifier.predict(temp)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
		C = plt.contour(xx1, xx2, Z, 0, colors=color, alpha=alpha, linewidths=linewidths)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
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