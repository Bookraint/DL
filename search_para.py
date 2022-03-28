import numpy as np
from network.fc import FullyConnect
from network.softmax import Softmax
from network.relu import Relu
from network.model import model as Model
from myutils import util

images, labels = util.load_mnist('./mnist')
labels = util.convert_labels(labels, num_classes=10)
val_size = int(len(images)*0.2)
val_data,val_labels = images[:val_size],labels[:val_size]
train_data,train_labels = images[val_size:],labels[val_size:]




num_epoch = 10
learning_rate_decay = 0.001

batch_size = 64
hidden_size = 40
learning_rate = 1e-5
l2_para = 1e-5





#bacth_size
batch_size_to_choose = list(range(32,256,32))
batch_res = []
for batch_size in batch_size_to_choose:
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

    best_acc = 0
    step = 0
    print("batch_size: ",batch_size)
    for epoch in range(num_epoch):
        learning_rate = learning_rate * 1.0 / (1.0 + learning_rate_decay * epoch)
        for i in range(train_data.shape[0] // batch_size):
            step += 1
            img = train_data[i * batch_size:(i + 1) * batch_size]
            label = train_labels[i * batch_size:(i + 1) * batch_size]
            loss=model.fit(img,label,lr=learning_rate)
            if step%1000==0:
                val_loss,val_acc = val(model)
                best_acc = max(best_acc, val_acc)
                print('epoch:%s---step:%s---train_loss:%.5f------val_acc:%.5f'%(epoch,step,loss/batch_size,val_acc))


    batch_res.append(best_acc)
print(batch_res)
print("Best bacth_size is:  ",batch_size_to_choose[np.argmax(batch_res)])


# #learning rate
# lr_to_choose = np.linspace(1e-5,5e-5,num=5)
# lr_res = []
# for lr in lr_to_choose:
#     fc1=FullyConnect([batch_size,784],hidden_size,l2_para=l2_para)
#     relu=Relu(fc1.output_shape)
#     fc = FullyConnect(fc1.output_shape, 10,l2_para=l2_para)
#     sf = Softmax(fc.output_shape)
#     model=Model()
#     model.add(fc1)
#     model.add(relu)
#     model.add(fc)
#     model.add(sf)



#     def val(model):
#         probs = model.predict(val_data)
#         loss = np.sum(-np.log(probs[range(probs.shape[0]),np.argmax(val_labels,axis=-1)]))/probs.shape[0]
#         pred = np.argmax(probs, axis=-1)
#         acc = np.sum(pred==np.argmax(val_labels,axis=-1))/pred.shape[0]
#         return loss, acc


#     best_acc = 0
#     step = 0
#     print("lr: ",lr)
#     for epoch in range(num_epoch):
#         learning_rate = learning_rate * 1.0 / (1.0 + learning_rate_decay * epoch)
#         for i in range(train_data.shape[0] // batch_size):
#             step += 1
#             img = train_data[i * batch_size:(i + 1) * batch_size]
#             label = train_labels[i * batch_size:(i + 1) * batch_size]
#             loss=model.fit(img,label,lr=learning_rate)
#             if step%1000==0:
#                 val_loss,val_acc = val(model)
#                 best_acc = max(best_acc, val_acc)
#                 print('epoch:%s---step:%s---train_loss:%.5f------val_acc:%.5f'%(epoch,step,loss/batch_size,val_acc))


#     lr_res.append(best_acc)
# print(lr_res)
# print("Best learning rate is:  ",lr_to_choose[np.argmax(lr_res)])


#hidden_size
# hidden_size_to_choose = list(range(20,120,10))
# hidden_size_res = []
# for hidden_size in hidden_size_to_choose:
#     fc1=FullyConnect([batch_size,784],hidden_size,l2_para=l2_para)
#     relu=Relu(fc1.output_shape)
#     fc = FullyConnect(fc1.output_shape, 10,l2_para=l2_para)
#     sf = Softmax(fc.output_shape)
#     model=Model()
#     model.add(fc1)
#     model.add(relu)
#     model.add(fc)
#     model.add(sf)



#     def val(model):
#         probs = model.predict(val_data)
#         loss = np.sum(-np.log(probs[range(probs.shape[0]),np.argmax(val_labels,axis=-1)]))/probs.shape[0]
#         pred = np.argmax(probs, axis=-1)
#         acc = np.sum(pred==np.argmax(val_labels,axis=-1))/pred.shape[0]
#         return loss, acc



#     best_acc = 0
#     step = 0
#     print("hidden_size: ",hidden_size)
#     for epoch in range(num_epoch):
#         learning_rate = learning_rate * 1.0 / (1.0 + learning_rate_decay * epoch)
#         for i in range(train_data.shape[0] // batch_size):
#             step += 1
#             img = train_data[i * batch_size:(i + 1) * batch_size]
#             label = train_labels[i * batch_size:(i + 1) * batch_size]
#             loss=model.fit(img,label,lr=learning_rate)
#             if step%1000==0:
#                 val_loss,val_acc = val(model)
#                 best_acc = max(best_acc, val_acc)
#                 print('epoch:%s---step:%s---train_loss:%.5f------val_acc:%.5f'%(epoch,step,loss/batch_size,val_acc))


#     hidden_size_res.append(best_acc)
# print(hidden_size_res)
# print("Best learning rate is:  ",hidden_size_to_choose[np.argmax(hidden_size_res)])

# #L2
# l2_to_choose = np.linspace(1e-5,5e-5,num=5)
# l2_res = []
# for l2_para in l2_to_choose:
#     fc1=FullyConnect([batch_size,784],hidden_size,l2_para=l2_para)
#     relu=Relu(fc1.output_shape)
#     fc = FullyConnect(fc1.output_shape, 10,l2_para=l2_para)
#     sf = Softmax(fc.output_shape)
#     model=Model()
#     model.add(fc1)
#     model.add(relu)
#     model.add(fc)
#     model.add(sf)



#     def val(model):
#         probs = model.predict(val_data)
#         loss = np.sum(-np.log(probs[range(probs.shape[0]),np.argmax(val_labels,axis=-1)]))/probs.shape[0]
#         pred = np.argmax(probs, axis=-1)
#         acc = np.sum(pred==np.argmax(val_labels,axis=-1))/pred.shape[0]
#         return loss, acc


#     best_acc = 0
#     step = 0
#     print("L2 regulation para: ",l2_para)
#     for epoch in range(num_epoch):
#         learning_rate = learning_rate * 1.0 / (1.0 + learning_rate_decay * epoch)
#         for i in range(train_data.shape[0] // batch_size):
#             step += 1
#             img = train_data[i * batch_size:(i + 1) * batch_size]
#             label = train_labels[i * batch_size:(i + 1) * batch_size]
#             loss=model.fit(img,label,lr=learning_rate)
#             if step%1000==0:
#                 val_loss,val_acc = val(model)
#                 best_acc = max(best_acc, val_acc)
#                 print('epoch:%s---step:%s---train_loss:%.5f------val_acc:%.5f'%(epoch,step,loss/batch_size,val_acc))


#     l2_res.append(best_acc)
# print(l2_res)
# print("Best L2 parameter is:  ",l2_to_choose[np.argmax(l2_res)])