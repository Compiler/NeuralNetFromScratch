import numpy as np
import math
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases

class Activation_RectifiedLinear:
    def forward(self, inputs):
        self.outputs=np.maximum(0, inputs)

class Activation_SoftMax:
    def activate(self, inputs, outputs):
        exp_vals = []
        for output in outputs:
            exp_vals.append(math.e**output)

        print(exp_vals)

        norm_vals=[]
        total = sum(exp_vals)
        for value in exp_vals:
            norm_vals.append(value / total)
        print(norm_vals)


inputs= [[1,2,3,2.5],
        [2,5,-1,2],
        [-1.5,2.7,3.3,-0.8]]
weights=[[0.2,0.8,-0.5,1],
        [0.5, -0.91, 0.26,-0.5],
        [-0.26,-0.27,0.17,0.87]]
biases=[2,3,0.5]
layer_outputs= np.dot(inputs, np.array(weights).T)+biases
print(layer_outputs)

fn = Activation_SoftMax()
fn.activate(inputs,layer_outputs[0])