# %%
import os
import nltk
import re
import pyprind
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import numpy as np

nltk.download("stopwords")

basepath = "/home/itz-amethyst/dev/ML_Space/ML_With_Pytorch_Scikit_learn_Sebastian/ch08/aclImdb"
labels = {"pos": 1, "neg": 0}
pybar = pyprind.ProgBar(50000, stream=sys.stdout)
rows = []
for s in ("test", "train"):
    for l in ("pos", "neg"):
        path = os.path.join(basepath, s, l)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file)) as infile:
                txt = infile.read()
            rows.append((txt, labels[l]))
            pybar.update()

df = pd.DataFrame(rows, columns=['review', 'sentiment'])
print(os.listdir(basepath))

# %%
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv("/home/itz-amethyst/dev/ML_Space/ML_With_Pytorch_Scikit_learn_Sebastian/ch08/movie_data.csv", index=False)
df.head()
df = pd.read_csv("/home/itz-amethyst/dev/ML_Space/ML_With_Pytorch_Scikit_learn_Sebastian/ch08/movie_data.csv")
df.shape
# Not needed
# df = df.rename(columns={"0": "review", "1": "sentiment"})
# df.head(3)

# %%
count = CountVectorizer()
docs = np.array(
    [
        "The sun is shining",
        "The weather is sweet",
        "The sun is shining, the weather is sweet, and one and one is two",
    ]
)
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# %%
def preprocess(text):
    text = re.sub("<[^>]*", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return text


df.loc[9, "review"][-50:]
preprocess(df.loc[9, "review"][-50:])
preprocess("</a>this :) is :( a test :-)!")
df["review"] = df["review"].apply(preprocess)

# %%
porter = PorterStemmer()
stop = stopwords.words("english")


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    return text.split()


txt = "runners like running and thus they run and so on"
[w for w in tokenizer_porter(txt) if w not in stop]
# tokenizer_porter(txt)

# %%
# X_train = df.iloc[:25000]["review"].values
# X_test = df.iloc[25000:]["review"].values
# y_test = df.iloc[25000:]["sentiment"].values
# y_train = df.iloc[:25000]["sentiment"].values
X_train = df.iloc[:25000]["review"].to_numpy(dtype=str)
X_test = df.iloc[25000:]["review"].to_numpy(dtype=str)
y_test = df.iloc[25000:]["sentiment"].to_numpy(dtype=int)

y_train = df.iloc[:25000]["sentiment"].to_numpy(dtype=int)
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

small_param_grid = [
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [None],
        "vect__tokenizer": [tokenizer, tokenizer_porter],
        "clf__penalty": ["l2"],
        "clf__C": [1.0, 10.0],
    },
    {
        "vect__ngram_range": [(1, 1)],
        "vect__stop_words": [stop, None],
        "vect__tokenizer": [tokenizer],
        "vect__use_idf": [False],
        "vect__norm": [None],
        "clf__penalty": ["l2"],
        "clf__C": [1.0, 10.0],
    },
]
lr_tdidf = Pipeline([("vect", tfidf), ("clf", LogisticRegression(solver="liblinear"))])
gs_lr_tfidf = GridSearchCV(
    lr_tdidf, small_param_grid, scoring="accuracy", cv=5, verbose=2, n_jobs=-1
)
gs_lr_tfidf.fit(X_train, y_train)
print(f"Best parameters set: {gs_lr_tfidf.best_params_}")
print(f"CV accuracy: {gs_lr_tfidf.best_score_:.3f}")
clf = gs_lr_tfidf.best_estimator_
print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")


# %%
stop = stopwords.words("english")
def tokenizer(text):
    text = re.sub("<[^>]*", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized 

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # to skip the header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

next(stream_docs(path="/home/itz-amethyst/dev/ML_Space/ML_With_Pytorch_Scikit_learn_Sebastian/ch08/movie_data.csv"))

# %%
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log_loss', random_state=1) # we initialized logistic regression classifier by setting loss to log
doc_stream = stream_docs(path='/home/itz-amethyst/dev/ML_Space/ML_With_Pytorch_Scikit_learn_Sebastian/ch08/movie_data.csv')

# %%
pbar = pyprind.ProgBar(45, stream=sys.stdout)
classes = np.array([0,1])
# do not run this twice unless we get ran out of documents to process for test stage
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
clf = clf.partial_fit(X_test, y_test)

# %%
df = pd.read_csv('/home/itz-amethyst/dev/ML_Space/ML_With_Pytorch_Scikit_learn_Sebastian/ch08/movie_data.csv')
df.head()
count = CountVectorizer(stop_words='english', max_df=.1,max_features=5000)
X = count.fit_transform(df['review'].values)

# %%
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)
lda.components_.shape

n_top_words = 5
feature_names = count.get_feature_names_out()
# will print the most important words from each topic 
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1:-1]]))

# horror category is topic number 6 with index 0 is number 5
horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
    print(f"Horror Movie #{(iter_idx + 1)}: \n")
    print(df['review'][movie_idx][:300], '...')
