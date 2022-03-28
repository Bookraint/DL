import pickle
class model(object):
    def __init__(self):
        self.layerlist=[]
        self.layernum=len(self.layerlist)
    def add(self,layer):
        self.layerlist.append(layer)
    def predict(self,x):
        out=x
        for layer in self.layerlist[:-1]:
            out=layer.forward(out)
        out=self.layerlist[-1].predict(out)
        return out
    def fit(self,x,y,lr=1e-5):
        out=x
        for layer in self.layerlist[:-1]:
            out=layer.forward(out)
        sflayer=self.layerlist[-1]
        loss=sflayer.cal_loss(out,y)
        ele=sflayer.gradient()
        self.layernum=len(self.layerlist)
        for i in range(self.layernum-2,-1,-1):
            ele=self.layerlist[i].gradient(ele)
        for layer in self.layerlist:
            if layer.layername == 'fc':
                layer.backward(lr=lr)
        return loss

    def save(self,name):
        file = open(name, 'wb')
        pickle.dump(self.layerlist,file)
        file.close()
    def load(self,name):
        file = open(name, 'rb')
        self.layerlist=pickle.load(file)
        file.close()
