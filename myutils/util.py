import numpy as np 
from glob import glob
import struct

def convert_labels(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    n = y.shape[0]
    category= np.zeros((n, num_classes))
    category[np.arange(n), y] = 1
    return category

def load_mnist(path, mode='train'):
    images_path = glob('./%s/%s*3-ubyte' % (path, mode))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, mode))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels