# %%
from tensorflow.python.tpu.tpu import shard
from ray.util.client.common import return_refs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ML_With_Pytorch_Scikit_learn_Sebastian.ch11.neuralnet import NeuralNetMLP

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.values
y = y.astype(int).values
print(X.shape)
print(y.shape)

# TO range it from -1 to 1
X = ((X / 255.0) - 0.5) * 2


# %%
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    image = X[y == i][0].reshape(28, 28)
    ax[i].imshow(image, cmap="Grays")

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):
    image = X[y == 6][i].reshape(28, 28)
    ax[i].imshow(image, cmap="Grays")

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


# %%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

# %%
model = NeuralNetMLP(num_features=28 * 28, num_hidden=50, num_classes=10)

num_epochs = 50
minibatch_size=100

def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx: start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

for i in range(num_epochs):
    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

    for X_train_mini , y_train_mini in minibatch_gen:
        break
    break

print(X_train_mini.shape)
print(y_train_mini.shape)



