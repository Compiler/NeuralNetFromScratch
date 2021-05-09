import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases

class Activation_RectifiedLinear:
    def forward(self, inputs):
        self.output=np.maximum(0, inputs)

class Activation_SoftMax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_vals / np.sum(exp_vals, axis=1,keepdims=True)
        self.output = probabilities


data = [[]]
for i in range(50):
    for x in range(10):
        data.append(math.sin(np.random.randn()))
    #print(data[i])

# x,y = spiral_data(samples=100, classes=3);
# print(x,'\n',y)
dense1 = DenseLayer(1, 5)
act1 = Activation_RectifiedLinear()
dense2 = DenseLayer(5, 2)
oact = Activation_SoftMax()

dense1.forward(data)
act1.forward(dense1.output)
dense1.forward(act1.output)
oact.forward(dense2.output)
print(oact.output)