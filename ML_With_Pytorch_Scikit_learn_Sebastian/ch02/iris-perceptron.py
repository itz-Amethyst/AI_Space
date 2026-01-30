# %%
from ML_Space.ML_With_Pytorch_Scikit_learn_Sebastian.ch02.adaline import AdalineGD
import numpy as np
import matplotlib.pyplot as plt
from ML_With_Pytorch_Scikit_learn_Sebastian.ch02.perceptron_np import Perceptron
from ML_With_Pytorch_Scikit_learn_Sebastian.ch02.plot_util import plot_decision_regions
import os
import pandas as pd


# %%
s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(s, header=None)
df.tail()


# %%
# feature column 5 which is the y
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)
# extract sepal length and petal length (feature 1 and 3)
X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="s", label="Versicolor")
plt.xlabel("Sepal length")
plt.xlabel("Petal length")
plt.legend(loc="upper left")
plt.show();

# %%
ptn = Perceptron(eta=0.1, n_iter=10)
ptn.fit(X, y)
plt.plot(range(1, len(ptn.errors_) + 1), ptn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.show();

# %%
plot_decision_regions(X, y, classifier=ptn)
plt.xlabel("Sepal length")
plt.xlabel("Petal length")
plt.legend(loc="upper left")
plt.show();

# %%
# standardize the dataset
X_std = np.copy(X)
X_std[:,0] = (X[:,0]- X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1]- X[:,1].mean()) / X[:,1].std()

ada_gd = AdalineGD(n_iter=20, eta=0.5)
ada_gd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title("Adaline- Gradient Descent")
plt.xlabel("Sepal length")
plt.xlabel("Petal length")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show();

plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.tight_layout()
plt.show();

 

# %%
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show();
