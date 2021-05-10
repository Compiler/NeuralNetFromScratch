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

	
class DenseLayer:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases=np.zeros((1,n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights)+self.biases
		print(self.output)

class Activation_RectifiedLinear:
	def forward(self, inputs):
		self.output=np.maximum(0, inputs)

class Activation_SoftMax:
	def forward(self, inputs):
		exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_vals / np.sum(exp_vals, axis=1,keepdims=True)
		self.output = probabilities


data = []
for i in range(500):
	data.append([math.sin(np.random.randn()), math.cos(np.random.randn()),])
print(data)
dense1 = DenseLayer(2, 1)
act1 = Activation_RectifiedLinear()
dense2 = DenseLayer(1, 2)
oact = Activation_SoftMax()

dense1.forward(data)
act1.forward(dense1.output)
dense2.forward(act1.output)
oact.forward(dense2.output)
print(oact.output)



x,y = create_data(samples=100, classes=3);
print(x,'\n',y)
dense1 = DenseLayer(2, 3)
act1 = Activation_RectifiedLinear()
dense2 = DenseLayer(3, 3)
oact = Activation_SoftMax()

dense1.forward(x)
act1.forward(dense1.output)
dense2.forward(act1.output)
oact.forward(dense2.output)
print(oact.output)