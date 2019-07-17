import numpy as np

# Attention, y需要是-1和1
class AdaBooster:
	def __init__(self, base, T):
		self.base = base
		self.T    = T
		self.U_   = []

	def fit(self, X, y):
		self.u_ = []
		m = X.shape[0]
		n = X.shape[1]
		u = np.ones(m)/m
		self.U_.append(u.sum())
		g_classifier = []
		g_alpha      = []
		for t in range(self.T):
			g = (self.base)()
			g.fit(X, y, u=u)
			g_classifier.append(g)
			# calculate the weighted error rate
			weighted_error_rate = g.error_rate_
			# calculate the scaler
			scaler = np.sqrt((1-weighted_error_rate)/weighted_error_rate)
			# adjust u
			y_predict = g.predict(X)
			u[y_predict == y] /= scaler
			u[y_predict != y] *= scaler
			self.U_.append(u.sum())
			# calculate alpha
			alpha = np.log(scaler)
			g_alpha.append(alpha)
		self.g_classifier_ = g_classifier
		self.g_alpha_      = g_alpha
		return self

	def predict(self, X):
		voting_basket = 0
		for i,alpha in zip(self.g_classifier_,self.g_alpha_):
			voting_basket += alpha * (i.predict(X))
		result = np.sign(voting_basket)
		result = np.where(result == 0, -1, result)
		return result