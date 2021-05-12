import numpy as np
import math


class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output,y)
		data_loss = np.mean(sample_losses)
		return data_loss


class Loss_CategoricalCrossEntropy(Loss):
	def forward(self, y_hat, y):
		samples = len(y_hat)

		y_hat_clipped = np.clip(y_hat, 1e-7,1-1e-7)
		if len(y.shape) == 1:
			correctConfidences = y_hat_clipped[range(samples), y]
		elif len(y.shape) == 2:
			correctConfidences = np.sum(y_hat_clipped * y, axis=1)

		return -np.log(correctConfidences)
	
class DenseLayer:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases=np.zeros((1,n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights)+self.biases

class Activation_RectifiedLinear:
	def forward(self, inputs):
		self.output=np.maximum(0, inputs)

class Activation_SoftMax:
	def forward(self, inputs):
		exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_vals / np.sum(exp_vals, axis=1,keepdims=True)
		self.output = probabilities



x = [1, -2, 3]
w = [-3, -1, 2]
b = 1.0

xw = [x[0] * w[0], x[1] * w[1], x[2] * w[2]]
print(xw)

z = np.sum(xw) + b
print(z)

y = max(z, 0) #relu
print(y)