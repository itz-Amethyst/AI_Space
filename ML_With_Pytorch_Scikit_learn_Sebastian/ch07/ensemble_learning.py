# %%
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb
from ML_With_Pytorch_Scikit_learn_Sebastian.ch07.mj_classifier import (
    MajorityVoteClassifier,
)
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from itertools import product
import math


# %%
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [
        comb(n_classifier, k) * error**k * (1 - error) ** (n_classifier - k)
        for k in range(k_start, n_classifier + 1)
    ]
    return sum(probs)


ensemble_error(n_classifier=11, error=0.25)
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]
plt.plot(error_range, ens_errors, label="Ensemble error", linewidth=2)
plt.plot(error_range, error_range, linestyle="--", label="Base error", linewidth=2)
plt.xlabel("Base error")
plt.ylabel("Base/Ensemble error")
plt.legend(loc="upper right")
plt.grid(alpha=0.5)
plt.show()
# %%

np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))

ex = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
np.argmax(p)

# %%
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y
)

# %%
clf1 = LogisticRegression(penalty="l2", C=1e-3, solver="lbfgs", random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")
pipe1 = Pipeline([["sc", StandardScaler()], ["clf", clf1]])
# DecisionTreeClassifier does not requires to preprocess the data
pipe3 = Pipeline([["sc", StandardScaler()], ["clf", clf3]])
clf_labels = ["Logistic regression", "Decision tree", "KNN"]
print("10 Fold corss validation \n")
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
    )
    print(f"ROC AUC: {scores.mean():.2f}")
    print(f"(+/-: {scores.std():.2f}) [{label}]")


# %%
# Using MajorityVoteClassifier
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels.append("Majority voting")
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(
        estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc"
    )
    print(f"ROC AUC: {scores.mean():.2f}")
    print(f"(+/-: {scores.std():.2f}) [{label}]")

colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label=f"{label} (auc = {roc_auc:.2f})")
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel("False positive rate (FPR)")
plt.ylabel("True positive rate (TPR)")
plt.show()

# %%
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(
        X_train_std[y_train == 0, 0],
        X_train_std[y_train == 0, 1],
        c="blue",
        marker="^",
        s=50,
    )
    axarr[idx[0], idx[1]].scatter(
        X_train_std[y_train == 1, 0],
        X_train_std[y_train == 1, 1],
        c="green",
        marker="o",
        s=50,
    )
    axarr[idx[0], idx[1]].set_title(tt)
    plt.text(
        -3.5,
        -5.0,
        s="Sepal width [standardized]",
        ha="center",
        va="center",
        fontsize=12,
    )
plt.text(
    -12.5,
    4.5,
    s="Petal length [standardized]",
    ha="center",
    va="center",
    fontsize=12,
    rotation=90,
)
plt.show()

# %%
# Getting best parameter settings with GridSearch
mv_clf.get_params()
params = {
    "decisiontreeclassifier__max_depth": [1, 2],
    "pipeline-1__clf__C": [0.001, 0.01, 100.0],
}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring="roc_auc")
# no need
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
    mean_score = grid.cv_results_["mean_test_score"][r]
    std_test = grid.cv_results_["std_test_score"][r]
    params = grid.cv_results_["params"][r]

    print(f"{mean_score:.3f} +/- {std_test:.2f} {params}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"ROC AUC: {grid.best_score_:.2f}")


# %%
# Bagging
df_wine = datasets.load_wine(as_frame=True)
df_wine.frame
df_wine = pd.DataFrame(df_wine.frame)
# to drop first column which is the id
df_wine.head()

df_wine = df_wine[df_wine["target"] != 0]
y = df_wine["target"].values
X = df_wine[["alcohol", "od280/od315_of_diluted_wines"]]
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)


# %%
tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=None)
bag = BaggingClassifier(
    estimator=tree,
    n_estimators=500,
    max_samples=1.0,
    max_features=1.0,
    bootstrap=True,
    bootstrap_features=False,
    n_jobs=-1,
    random_state=1,
)

# Without bagging applied (overfitting occurred)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Decision tree train/test score:  {tree_train:.3f}, {tree_test:.3f}")

# With bagging
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f"Baggin tree train/test score:  {tree_train:.3f}, {tree_test:.3f}")

# %%
x_min = X_train.iloc[:, 0].min() - 1
x_max = X_train.iloc[:, 0].max() + 1
y_min = X_train.iloc[:, 1].min() - 1
y_max = X_train.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex="col", sharey="row", figsize=(8, 3))

for idx, clf, tt in zip([0, 1], [tree, bag], ["Decision Tree", "Bagging"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(
        X_train.iloc[y_train == 0, 0],
        X_train.iloc[y_train == 0, 1],
        c="blue",
        marker="^",
    )
    axarr[idx].scatter(
        X_train.iloc[y_train == 1, 0],
        X_train.iloc[y_train == 1, 1],
        c="green",
        marker="o",
    )
    axarr[idx].set_title(tt)
axarr[0].set_ylabel("Alcohol", fontsize=12)
plt.tight_layout()
plt.text(
    0,
    -0.2,
    s="OD280/OD315 of diluted wines",
    ha="center",
    va="center",
    fontsize=12,
    transform=axarr[1].transAxes,
)
plt.show()
# %%
# AdaBoost with numpy

y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
correct = y == yhat
# It's like softmax telling that we have 10 items of 0.1 sum of them obviously should be 10
weights = np.full(10, 0.1)
print(weights)
# note that putt
~correct
correct
# Basically ~ before the correct, inverts the values like true -> false and false -> true
epsilon = np.mean(~correct)
print(epsilon)

alpha_j = 0.5 * np.log((1 - epsilon) / epsilon)
update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
update_if_wrong_1 = 0.1 * np.exp(-alpha_j * 1 * -1)
weights = np.where(correct == 1, update_if_correct, update_if_wrong_1)
normalized_weights = weights / np.sum(weights)
print(normalized_weights)
print(weights)


# %%
# Adaboost with scikit-learn
tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=1)
ada = AdaBoostClassifier(
    estimator=tree, n_estimators=500, learning_rate=0.1, random_state=1
)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(
    f"Decision tree train/test accuracies (without adaboost) : {tree_train:.3f} / {tree_test:.3f}"
)

# With adaboost applied
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(
    f"Adaboost train/test accuracies (without adaboost) : {tree_train:.3f} / {tree_test:.3f}"
)

# %%
x_min = X_train.iloc[:, 0].min() - 1
x_max = X_train.iloc[:, 0].max() + 1
y_min = X_train.iloc[:, 1].min() - 1
y_max = X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2, sharex="col", sharey="row", figsize=(8, 3))
for idx, clf, tt in zip([0, 1], [tree, ada], ["Decision tree", "AdaBoost"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(
        X_train.iloc[y_train == 0, 0], X_train.iloc[y_train == 0, 1], c="blue", marker="^"
    )
    axarr[idx].scatter(
        X_train.iloc[y_train == 1, 0], X_train.iloc[y_train == 1, 1], c="green", marker="o"
    )
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel("Alcohol", fontsize=12)
plt.tight_layout()
plt.text(
    0,
    -0.2,
    s="OD280/OD315 of diluted wines",
    ha="center",
    va="center",
    fontsize=12,
    transform=axarr[1].transAxes,
)
plt.show()

# %%
# GradientBoosting with XGBoost
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=1, use_label_encoder=False)
gbm = model.fit(X_train, y_train)
y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)

gbm_train = accuracy_score(y_train, y_train_pred)
gbm_test = accuracy_score(y_test, y_test_pred)
print(
    f"XGBoost train/test accuracies  : {gbm_train:.3f} / {gbm_test:.3f}"
)
