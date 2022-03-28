import numpy as np
from functools import reduce

class FullyConnect(object):
    def __init__(self, input_shape, output_size,l2_para):
        self.input_shape = [int(x) for x in input_shape]
        input_size = input_shape[1]
        self.output_shape = [input_shape[0], output_size]
        self.weights = np.random.randn(input_size, output_size)/100
        self.bias = np.random.randn(output_size)/100
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.l2_para = l2_para
        self.layername = "fc"

    def forward(self, x):
        self.x=x
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, ele):
        for i in range(ele.shape[0]):  ##batch_size
            col_x = self.x[i][:, np.newaxis]
            ele_i = ele[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, ele_i) + self.l2_para*self.weights
            self.b_gradient += ele_i.reshape(self.bias.shape) + self.l2_para*self.bias

        next_ele = np.dot(ele, self.weights.T)
        next_ele = np.reshape(next_ele, self.input_shape)
        return next_ele

    def backward(self, lr=0.00001):
        self.weights -= lr * self.w_gradient
        self.bias -= lr * self.bias
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
