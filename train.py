from BayesClassification import train_bayes, save_model
from Data import load_dataset

if __name__ == "__main__":
    X_train, y_train = load_dataset('./Data/ProcessedData/', 3000)
    model = train_bayes(X_train, y_train, vocab_size=500)
    print("Vocabulary:", model['vocab'])
    save_model(model, './Model/bayes.npz')
