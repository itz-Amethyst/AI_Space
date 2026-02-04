# %%
from tensorflow.python.tpu.tpu import shard
from ray.util.client.common import return_refs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from ML_With_Pytorch_Scikit_learn_Sebastian.ch11.neuralnet import NeuralNetMLP
from ML_With_Pytorch_Scikit_learn_Sebastian.ch11.utils import int_to_onehot

# ----------- loading from openml -----------------
# X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
# ----------- loading from openml -----------------

dataset = load_dataset('ylecun/mnist')
def hf_to_numpy(ds):
    X = np.stack([np.array(sample['image'], dtype=np.float32).reshape(-1) for sample in ds])
    y = np.array([sample['label'] for sample in ds], dtype=np.int64)
    return X,y
# ----------- loading from openml -----------------
# X = X.values
# y = y.astype(int).values
# ----------- loading from openml -----------------
X_train_hf, y_train_hf = hf_to_numpy(dataset['train'])
X_test_hf, y_test_hf = hf_to_numpy(dataset['test'])
X = np.concatenate([X_train_hf, X_test_hf], axis=0)
y = np.concatenate([y_train_hf, y_test_hf], axis=0)
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
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
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


# %%
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse/ i
    acc = correct_pred / num_examples
    return mse, acc

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc*100:.1f}%')


# %%
def train(model, X_train, y_train, X_valid, y_valid, num_epochs,learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(
        X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            #### Compute outputs ####
            a_h, a_out = model.forward(X_train_mini)
            #### Compute gradients ####
            d_loss__d_w_out, d_loss__d_b_out, \
            d_loss__d_w_h, d_loss__d_b_h = \
            model.backward(X_train_mini, a_h, a_out,
            y_train_mini)
            #### Update weights ####
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out
        #### Epoch Logging ####
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train
        )
        valid_mse, valid_acc = compute_mse_and_acc(
        model, X_valid, y_valid
        )
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
        f'| Train MSE: {train_mse:.2f} '
        f'| Train Acc: {train_acc:.2f}% '
        f'| Valid Acc: {valid_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc

np.random.seed(123)
epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=0.1)



# %%
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel("Mean squared error")
plt.xlabel("Epoch")
plt.show();

plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(loc='lower right')
plt.show();

# %%
test_mes, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')
X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]
_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas,axis=1)
misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8,8))

ax = ax.flatten()
for i in range(25):
    img = misclassified_images[i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) ' f'True: {correct_labels[i]}\n' f'Predicted: {misclassified_labels[i]}')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show();

# %%

