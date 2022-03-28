import matplotlib.pyplot as plt
import numpy as np
train_loss_set,val_loss_set,val_acc_set = np.load("to_plot.npy")
n = len(train_loss_set)

plt.figure()
plt.plot(range(n),train_loss_set)
plt.xlabel("train step")
plt.title("Train loss");
plt.savefig("./output/train_loss.png")

plt.clf()
plt.figure()
plt.plot(range(n),val_loss_set)
plt.xlabel("train step")
plt.title("Val loss");
plt.savefig("./output/val_loss.png")

plt.clf()
plt.figure()
plt.plot(range(n),val_acc_set)
plt.xlabel("train step")
plt.title("Val Acc");
plt.savefig("./output/val_acc.png")