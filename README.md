# DL
## 项目介绍
本项目利用numpy搭建两层神经网络对MNIST数据集进行分类
***
## 环境安装
numpy  
matplotlib
***
## 文件结构
```
│  plt.py       #用于可视化训练和测试loss曲线
│  search_para.py    #搜索超参
│  test.py      #测试集输出分类精度
│  to_plot.npy
│  train.py     #训练
│      
├─mnist    #数据集
│      
├─model_path    #模型存放途径
│      fc.pkl
├─myutils
│  │  util.py   #加载数据等函数
│  │  __init__.py
│          
├─network    #神经网络layer
│  │  fc.py    
│  │  model.py  
│  │  relu.py
│  │  softmax.py
│  │  __init__.py
│          
└─output    #可视化loss结果
        train_loss.png
        val_acc.png
        val_loss.png
```
***
## 使用说明  
---
### 训练模型
```
python train.py
```
· 训练过程中会保存在验证集上acc最高的模型，保存路径为./model_path  
· 训练过程中的train_loss,val_loss,val_acc会保存到```to_plot.npy```文件
运行
```
python plt.py
```
可在./output中得到可视化结果  

---

### 搜索超参数
运行```search_para.py```文件，但具体需要搜索哪一个参数需要在其中稍加修改，结果会打印出候选区间内的最佳参数。  

---
### 模型测试
```--path_to_load```表示模型存放的位置，将模型文件放入路径后，运行下列命令输出测试集上的准确率、
```
python test.py --path_to_load ./model_path/fc.pkl
```
---
### 模型&数据
model: 
链接：https://pan.baidu.com/s/1eiJH68pMzczMPVck0aH-eA?pwd=bk9h  
提取码：bk9h  
data:
链接：https://pan.baidu.com/s/1xcbGra5fEpOhfVepNo7Vpw?pwd=obrt  
提取码：obrt