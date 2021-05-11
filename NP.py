import numpy as np
import math





def create_data(samples, classes):
	X = np.zeros((samples*classes, 2))
	y = np.zeros(samples*classes, dtype='uint8')
	for class_number in range(classes):
		ix = range(samples*class_number, samples*(class_number+1))
		r = np.linspace(0.0, 1, samples)
		t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
		X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
		y[ix] = class_number
	return X, y


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



num_neurons = 30
classes = 3
X,y = create_data(samples=100, classes=classes);
dense1 = DenseLayer(2, num_neurons)
act1 = Activation_RectifiedLinear()
dense2 = DenseLayer(num_neurons, classes)
oact = Activation_SoftMax()

dense1.forward(X)
act1.forward(dense1.output)
dense2.forward(act1.output)
oact.forward(dense2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(oact.output, y)
print("Loss:", loss)


