import numpy as np

import os
from itertools import islice, takewhile
from functools import reduce
from collections import Counter, defaultdict

# --- 两个工具高阶函数 ---
def pipe(data, *funcs):
    return reduce(lambda d, f: f(d), funcs, data)


def map_reduce(mapper, reducer):
    return lambda data: reduce(reducer, map(mapper, data))


# 构建词汇表
def build_vocab(docs, max_size=None):
    df_counter = pipe(
        docs,
        lambda seq: map(lambda doc: {w:1 for w in set(doc)}, seq),
        lambda seq: reduce(
            lambda a,b: {k: a.get(k,0)+b.get(k,0) for k in a.keys() | b.keys()},
            seq,
            {}
        )
    )
    
    sorted_vocab = sorted(
        df_counter.keys(),
        key=lambda x: (-df_counter[x], x)
    )
    
    truncate = lambda: (w for i, w in enumerate(sorted_vocab) 
                      if max_size is None or i < max_size)
    
    return list(truncate())


def calc_idf(docs, vocab):
    doc_count = len(docs)
    df_counter = map_reduce(
        lambda doc: defaultdict(int, {w: 1 for w in set(doc)}),
        lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
    )(docs)

    idf = np.array([
        np.log((doc_count + 1) / (df_counter.get(w, 0) + 1))  # 采用加 1 平滑
        for w in vocab
    ])
    return idf


def doc2vec(doc, vocab, idf):
    tf = pipe(
        doc,
        lambda d: Counter(d),
        lambda cnt: [cnt.get(w, 0) / len(doc) for w in vocab]
    )
    return np.multiply(tf, idf)


def train_bayes(X_train, y_train, smooth=1.0, vocab_size=None):
    # 构建特征空间
    vocab = build_vocab(X_train, max_size=vocab_size)
    idf = calc_idf(X_train, vocab)

    # 计算类别先验
    classes, counts = np.unique(y_train, return_counts=True)
    prior = counts / counts.sum()

    # 计算条件概率
    def class_probability(c):
        class_docs = [doc for doc, label in zip(X_train, y_train) if label == c]
        vecs = [doc2vec(doc, vocab, idf) for doc in class_docs]
        total = sum(vecs)  # 向量相加
        return (total + smooth) / (total.sum() + smooth * len(vocab))

    likelihood = np.array([class_probability(c) for c in classes])
    return {
        'vocab': vocab,
        'idf': idf,
        'classes': classes,
        'prior': prior,
        'likelihood': likelihood
    }


def predict(model, X_test):
    vocab = model['vocab']
    idf = model['idf']

    def predict_single(doc):
        vec = doc2vec(doc, vocab, idf)
        log_probs = np.log(model['prior']) + np.sum(
            vec * np.log(model['likelihood']), axis=1
        )
        return model['classes'][np.argmax(log_probs)]

    return [predict_single(doc) for doc in X_test]


def save_model(model, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        filepath,
        vocab=np.array(model['vocab'], dtype=object),
        idf=model['idf'],
        classes=model['classes'],
        prior=model['prior'],
        likelihood=model['likelihood']
    )


def load_model(filepath):
    with np.load(filepath, allow_pickle=True) as data:
        vocab = data['vocab'].tolist()
        return {
            'vocab': vocab,
            'vocab_index': {word: idx for idx, word in enumerate(vocab)},
            'idf': data['idf'],
            'classes': data['classes'],
            'prior': data['prior'],
            'likelihood': data['likelihood']
        }
