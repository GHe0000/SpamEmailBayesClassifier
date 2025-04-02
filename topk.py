import numpy as np
from functools import reduce

from BayesClassification import *

def get_top_words(model, n=10):
    def process_class(class_idx):
        probs = model['likelihood'][class_idx]
        word_prob_pairs = zip(model['vocab'], probs)
        sorted_pairs = sorted(word_prob_pairs, key=lambda x: -x[1])
        return [(word, prob) for word, prob in sorted_pairs[:n]]
    return {
        cls: process_class(idx)
        for idx, cls in enumerate(model['classes'])
    }

if __name__ == "__main__":
    model = load_model("./Model/bayes.npz")
    top_words = get_top_words(model, n=10)
    for cls, words in top_words.items():
        print(f"\n===== 类别 {cls} 重要词 =====")
        for word, prob in words:
            print(f"{word}: {prob:.4f}")
