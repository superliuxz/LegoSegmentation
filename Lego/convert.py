import numpy as np

train_label = np.genfromtxt('20.rb.256x192.label.txt', delimiter=',')

x = np.concatenate([train_label, train_label], axis=-1)

x[:, :512][np.where(x[:, :512] == 200)] = 1
x[:, :512][np.where(x[:, :512] == 100)] = 0
x[:, 512:][np.where(x[:, 512:] == 200)] = 0
x[:, 512:][np.where(x[:, 512:] == 100)] = 1

train_label[np.where(train_label == 200)] = 1
train_label[np.where(train_label == 100)] = 1

y = x[:, :512] + x[:, 512:]
res = np.equal(y, train_label)

assert len(np.where(res==False)[0])==0
assert len(np.where(res==False)[1])==0

np.savetxt('convert.txt', x, fmt='%i', delimiter=',')