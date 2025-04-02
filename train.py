import os
from itertools import islice
from BayesClassification import *
from pathlib import Path


def load_dataset(data_dir, n):
    X, y = [], []

    spam_dir = os.path.join(data_dir, 'spam')
    for filename in islice(os.listdir(spam_dir), n):
        with open(os.path.join(spam_dir, filename), 'r', encoding='utf-8') as f:
            words = f.read().split()
            X.append(words)
            y.append("spam")

    ham_dir = os.path.join(data_dir, 'ham')
    for filename in islice(os.listdir(ham_dir), n):
        with open(os.path.join(ham_dir, filename), 'r', encoding='utf-8') as f:
            words = f.read().split()
            X.append(words)
            y.append("ham")

    return X, np.array(y)


def load_one_file(filename):
    X = []
    with open(filename, 'r', encoding='utf-8') as f:
        words = f.read().split()
        X.append(words)
    return X

if __name__ == "__main__":
    # 示例训练数据
    X_train, y_train = load_dataset('./Data/ProcessedData/', 100)
    model = train_bayes(X_train, y_train, vocab_size=200)
    print("Vocabulary:", model['vocab'])  # 输出按文档频率排序的词汇
    save_model(model, './Model/bayes.npz')  # 保存模型

    # # 加载测试数据
    X_test = load_one_file('./Data/ProcessedData/spam/123.txt')
    y_pred = predict(model, X_test)
    print("Prediction:", *y_pred)  # 输出预测结果

