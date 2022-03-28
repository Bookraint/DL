import numpy as np
class Relu(object):
    def __init__(self, shape):
        self.ele = np.zeros(shape)
        self.x = np.zeros(shape)
        self.layername = "relu"
        
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, ele):
        self.ele = ele
        self.ele[self.x<0]=0
        return self.ele
