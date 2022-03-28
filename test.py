import numpy as np
import argparse
from network.fc import FullyConnect
from network.softmax import Softmax
from network.relu import Relu
from network.model import model as Model
from myutils import util

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_to_load",
    default='./model_path/fc.pkl',
    type=str,
    help="Model path to load.",
)
args = parser.parse_args()

images, labels = util.load_mnist('./mnist')
test_images, test_labels = util.load_mnist('./mnist', 't10k')
labels = util.convert_labels(labels, num_classes=10)
test_labels = util.convert_labels(test_labels, num_classes=10)
##split images into train_data, val_data
val_size = int(len(images)*0.2)
val_data,val_labels = images[:val_size],labels[:val_size]
train_data,train_labels = images[val_size:],labels[val_size:]


def test(model):
    probs = model.predict(test_images)
    loss = np.sum(-np.log(probs[range(probs.shape[0]),np.argmax(test_labels,axis=-1)]))/probs.shape[0]
    pred = np.argmax(probs, axis=-1)
    acc = np.sum(pred==np.argmax(test_labels,axis=-1))/pred.shape[0]
    return loss, acc



model = Model()
model.load(args.path_to_load)
loss, acc = test(model)
print("Accuracy on test:  %.5f"%(acc))