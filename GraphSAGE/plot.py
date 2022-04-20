import pickle
import matplotlib.pyplot as plt

import json 
train_losses = json.load(open('train_loss_10000_round_32.json', 'r'))
val_losses = json.load(open('val_loss_10000_round_32.json', 'r'))
train_losses = eval(train_losses)
train_losses = [float(i) for i in train_losses]
val_losses = eval(val_losses)
val_losses = [float(i) for i in val_losses]
for i in train_losses:
    print(train_losses)
import numpy as np
x = np.linspace(0, 10000, 201)
plt.plot(x, train_losses, label = "train_loss_emb32")
plt.plot(x, val_losses, label = "val_loss_emb32")

train_losses = json.load(open('train_loss_10000round_64.json', 'r'))
val_losses = json.load(open('val_loss_10000round_64.json', 'r'))
train_losses = eval(train_losses)
train_losses = [float(i) for i in train_losses]
val_losses = eval(val_losses)
val_losses = [float(i) for i in val_losses]

plt.plot(x, train_losses, label = "train_loss_emb64")
plt.plot(x, val_losses, label = "val_loss_emb64")


train_losses = json.load(open('train_loss_10000round_16.json', 'r'))
val_losses = json.load(open('val_loss_10000round_16.json', 'r'))
train_losses = eval(train_losses)
train_losses = [float(i) for i in train_losses]
val_losses = eval(val_losses)
val_losses = [float(i) for i in val_losses]

plt.plot(x, train_losses, label = "train_loss_emb16")
plt.plot(x, val_losses, label = "val_loss_emb16")
plt.legend()
plt.title('5000 round size 32 plot')
plt.show()