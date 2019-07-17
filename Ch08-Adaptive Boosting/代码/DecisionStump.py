import numpy as np

class DecisionStumpClassifier:
	def __init__(self):
		self.i_     = None
		self.theta_ = None
		self.s_     = None

	def fit(self, X, y, u=None):
		m = X.shape[0]
		n = X.shape[1]
		# if u is not given, then uniform
		if u is None:
			u = np.ones(m)
		total_u      = u.sum()
		total_u_half = total_u/2
		# initialize the variables for best parameters
		best_i       = 0
		best_theta   = 0
		best_s       = 0
		best_error_u = np.inf
		# iterations on each dimension
		for i in range(n):
			sorted_value = np.sort(X[:,i])
			mid_point    = (sorted_value[:-1]+sorted_value[1:])/2
			# iterations on each mid point as threshold theta
			for theta in mid_point:
				s = -1
				result = self.predict_inner(X, i, theta, s)
				# sum of u: for incorrect point, should be minimized
				sum_u_incorrect = u[result != y].sum()
				# if the sum of incorrect points' u > 1/2 sum of u
				# then changing direction will get a smaller sum of incorrect points' u
				# less than 1/2 sum of u
				if sum_u_incorrect > total_u_half:
					s = +1
					sum_u_incorrect = total_u - sum_u_incorrect
				if sum_u_incorrect < best_error_u:
					best_i       = i
					best_theta   = theta
					best_s       = s
					best_error_u = sum_u_incorrect	
		self.i_     = best_i
		self.theta_ = best_theta
		self.s_     = best_s
		# error_rate can be weighted or not
		error_rate = best_error_u/total_u
		self.error_rate_ = error_rate
		return self

	def predict(self, X):
		result = self.predict_inner(X, self.i_, self.theta_, self.s_)
		return result

	def predict_inner(self, X, i, theta, s):
		m = X.shape[0]
		n = X.shape[1]
		# if X is not a 2D-array, i.e. a vector, convert it
		if len(X.shape) == 1:
			X = X.reshape(1,n)
		result = s * np.sign(X[:,i] - theta)
		result = np.where(result==0, -1, result)
		return result
	
