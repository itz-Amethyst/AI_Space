# %%
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_digits
from ML_With_Pytorch_Scikit_learn_Sebastian.ch02.plot_util import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

df_wine = load_wine(as_frame=True)
df_wine.frame
df_wine = pd.DataFrame(df_wine.frame)
df_wine.head()


# %%
X, y = df_wine.iloc[:, :-1].values, df_wine.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

stdc = StandardScaler()
X_train_std = stdc.fit_transform(X_train)
X_test_std = stdc.transform(X_test)
len(X_test_std)

# To compute covariance matrix
cov_mat = np.cov(X_train_std.T)
# To perform eigendecomposition
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("EigenValues \n", eigen_vals)

# %%
total = sum(eigen_vals)
var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 14), var_exp, align="center", label="Individual explained variance")
plt.step(range(1, 14), cum_var_exp, where="mid", label="Cumulative explained variance")
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component index")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
# %%
# An alternative way of getting variance ratio is like this
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# %%
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
eigen_pairs

# We select the two largest eigenvalues
# output: 13 x 2 dimensional projection matrix
W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print("Matrix W: \n", W)
X_train_std[0].dot(W)
# Output 124 x 2
X_train_pca = X_train_std.dot(W)


# %%
colors = ["r", "b", "g"]
markers = ["o", "s", "^"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_pca[y_train == l, 0],
        X_train_pca[y_train == l, 1],
        c=c,
        label=f"Class {l}",
        marker=m,
    )

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
# %%
pca = PCA(n_components=2)
# Instead of defining ovr in the multiclass parameter of logisticRegression it's suggested to follow this strategy
lr = OneVsRestClassifier(LogisticRegression(random_state=1, solver="lbfgs"))

# Compress the data with PCA
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
# For test data
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# %%
loadings = eigen_vecs * np.sqrt(eigen_vals)
fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[:-1], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()





# %%

# An alternative way by calculating from pca scikit-learn
sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots()
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(df_wine.columns[:-1], rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()

# # %% [markdown]
# # # Numpy implementation of the LDA (Supervised method)
#
# # %%
# np.set_printoptions(precision=4)
# mean_vecs = []
# for label in range(0, 3):
#     mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
#     print(f'MV {label}: {mean_vecs[label]} \n')
# # number of features
# d = 13
# S_W = np.zeros((d,d))
# for label, mv in zip(range(0,3), mean_vecs):
#     class_scatter = np.zeros((d,d))
#     for row in X_train_std[y_train == label]:
#         row, mv = row.reshape(d, 1), mv.reshape(d, 1)
#         class_scatter += (row-mv).dot((row-mv).T)
#     S_W += class_scatter
# print('Within-class scatter matrix:' f'{S_W.shape[0]}x{S_W.shape[1]}')
# print("Class label distribution:", np.bincount(y_train)[:])
#
# d = 13 # number of features
# S_W = np.zeros((d, d))
# for label,mv in zip(range(1, 4), mean_vecs):
#     class_scatter = np.cov(X_train_std[y_train==label].T)
#     S_W += class_scatter
# print('Scaled within-class scatter matrix: ' f'{S_W.shape[0]}x{S_W.shape[1]}')
#
# mean_overall = np.mean(X_train_std, axis=0)
# mean_overall = mean_overall.reshape(d, 1)
# d = 13 # number of features
# S_B = np.zeros((d, d))
# for i, mean_vec in enumerate(mean_vecs):
#     n = X_train_std[y_train == i + 1, :].shape[0]
#     mean_vec = mean_vec.reshape(d, 1) # make column vector
#     S_B += n * (mean_vec - mean_overall).dot(
#     (mean_vec - mean_overall).T)
# print('Between-class scatter matrix: '
# f'{S_B.shape[0]}x{S_B.shape[1]}')
#
# # %%
# eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
# eigen_pairs = sorted(eigen_pairs, key=lambda k:k[0], reverse=True)
# print("Eigenvalues in descending order: \n")
# for eigen_val in eigen_pairs:
#     print(eigen_val[0])


# %%
#  LDA via Scikit-learn
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = OneVsRestClassifier(LogisticRegression(random_state=1, solver='lbfgs'))
lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
# for test subset
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# %% [markdown]
# # T-SNE (Non linear dimension reduction)



# %%
digits = load_digits()
digits
fig, ax = plt.subplots(1,4)
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
plt.show()

y_digits = digits.target
X_digits = digits.data

# With this code we projected 64 dimensional dataset onto 2 dimensional space
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)


# %%
def plot_projection(x, colors):

    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0], x[colors == i, 1])

    for i in range(10):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext,ytext,str(i), fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
plot_projection(X_digits_tsne, y_digits)
plt.show()
