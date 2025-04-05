import numpy as np

from BayesClassification import load_model, doc2vec
from Data import load_dataset, find_word_list

def predict_log_probs(model, X_test):
    vocab = model['vocab']
    idf = model['idf']

    def predict_single(doc):
        vec = doc2vec(doc, vocab, idf)
        log_probs = np.log(model['prior']) + np.sum(
            vec * np.log(model['likelihood']), axis=1
        )
        return log_probs

    return model['classes'], np.array([predict_single(doc) for doc in X_test])


if __name__ == "__main__":
    X_test, y_test = load_dataset('./Data/ProcessedData/', 500)
    model = load_model("./Model/bayes.npz")
    classes, log_probs = predict_log_probs(model, X_test)
    diff = log_probs[:, 1] - log_probs[:, 0]
    print(np.argmax(diff))
    print(X_test[np.argmax(diff)])
    print(find_word_list(X_test[np.argmax(diff)], './Data/ProcessedData/', 500, 'spam'))
    print(np.argmin(diff))
    print(X_test[np.argmin(diff)])
    print(find_word_list(X_test[np.argmax(diff)], './Data/ProcessedData/', 500, 'ham'))
