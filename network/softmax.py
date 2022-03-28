import numpy as np
class Softmax(object):
    def __init__(self, shape):
        self.x = np.zeros(shape)
        self.ele = np.zeros(shape)
        self.layername = "softmax"

    def cal_loss(self, prediction, labels):
        self.label = labels
        self.predict(prediction)
        loss=0
        for i in range(prediction.shape[0]):
            loss+=np.log(np.sum(np.exp(prediction[i]))) - prediction[i, np.argmax(labels[i])]
        return loss

    def predict(self, x):
        self.x = np.zeros(x.shape)
        for i in range(x.shape[0]):
            self.x[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
        return self.x
    def gradient(self):
        self.ele = self.x.copy()
        self.ele[range(self.x.shape[0]), np.argmax(self.label,axis=-1)] -= 1
        return self.ele