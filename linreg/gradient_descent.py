import numpy as np
from matplotlib import pyplot as plt

class GradientDescent:
	def __init__(self, learning_rate=0.1, theta=np.zeros((1, 2)), n_iterations=1000, plot=False):
		self.theta = theta
		self.n_iterations = n_iterations
		self.mse = dict()
		self.learning_rate = learning_rate
		self.plot = plot
		pass

	def fit(self, X, y):
		m = len(X)
		X_bias = np.c_[np.ones((len(X), 1)), X]
		for i in range(self.n_iterations + 1):
			y_pred = self.theta.dot(X_bias.T)[0]

			if (self.plot == True and i % 100 == 0):
				plt.figure(1)
				plt.plot(self.theta[0][1] * np.arange(2) + self.theta[0][0])
				plt.scatter(X, y)
				plt.xlabel('km_scaled')
				plt.ylabel('price')

			self.mse[i] = (1. / m) * sum((y - y_pred) ** 2)
			gradients = (-2. / m) * np.dot((y - y_pred).T, X_bias)
			step_size = self.learning_rate * gradients
			self.theta = self.theta - step_size
			pass

		return self
		pass

	def predict(self, X):
		X_bias = np.c_[np.ones((len(X), 1)), X]
		return np.dot(self.theta, X_bias.T)[0]

	def mean_squared_error(self, y, y_pred):
		return sum((y - y_pred) ** 2) / len(y)

	def mae(self, y, y_pred):
		return np.abs(y - y_pred).mean()
