import numpy as np
from network.fc import FullyConnect
from network.softmax import Softmax
from network.relu import Relu
from network.model import model as Model
from myutils import util

images, labels = util.load_mnist('./mnist')
test_images, test_labels = util.load_mnist('./mnist', 't10k')
labels = util.convert_labels(labels, num_classes=10)
test_labels = util.convert_labels(test_labels, num_classes=10)
##split images into train_data, val_data
val_size = int(len(images)*0.2)
val_data,val_labels = images[:val_size],labels[:val_size]
train_data,train_labels = images[val_size:],labels[val_size:]

batch_size = 64
hidden_size = 90
num_epoch = 10
learning_rate = 1e-5
learning_rate_decay = 0.001
l2_para = 5e-5

fc1=FullyConnect([batch_size,784],hidden_size,l2_para=l2_para)
relu=Relu(fc1.output_shape)
fc = FullyConnect(fc1.output_shape, 10,l2_para=l2_para)
sf = Softmax(fc.output_shape)
model=Model()
model.add(fc1)
model.add(relu)
model.add(fc)
model.add(sf)



def val(model):
    probs = model.predict(val_data)
    loss = np.sum(-np.log(probs[range(probs.shape[0]),np.argmax(val_labels,axis=-1)]))/probs.shape[0]
    pred = np.argmax(probs, axis=-1)
    acc = np.sum(pred==np.argmax(val_labels,axis=-1))/pred.shape[0]
    return loss, acc

def test(model):
    probs = model.predict(test_images)
    loss = np.sum(-np.log(probs[range(probs.shape[0]),np.argmax(test_labels,axis=-1)]))/probs.shape[0]
    pred = np.argmax(probs, axis=-1)
    acc = np.sum(pred==np.argmax(test_labels,axis=-1))/pred.shape[0]
    return loss, acc

best_acc = 0
step = 0
train_loss_set,val_loss_set,val_acc_set = [],[],[]

for epoch in range(num_epoch):
    learning_rate = learning_rate * 1.0 / (1.0 + learning_rate_decay * epoch)
    for i in range(train_data.shape[0] // batch_size):
        step += 1
        img = train_data[i * batch_size:(i + 1) * batch_size]
        label = train_labels[i * batch_size:(i + 1) * batch_size]
        loss=model.fit(img,label,lr=learning_rate)
        if step%100==0:
            val_loss,val_acc = val(model)
            train_loss_set.append(loss/batch_size)
            val_loss_set.append(val_loss)
            val_acc_set.append(val_acc)
            print('epoch:%s---step:%s---train_loss:%.5f-------val_loss:%.5f---val_acc:%.5f'%(epoch,step,loss/batch_size,val_loss,val_acc))
            if val_acc > best_acc:
                best_acc = val_acc
                print("save model...")
                model.save('./model_path/fc.pkl')

np.save("to_plot.npy",[train_loss_set,val_loss_set,val_acc_set])
