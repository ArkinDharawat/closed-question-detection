import pickle
import numpy as np
from sklearn.utils import class_weight
from collections import Counter

FOLDER_PATH = "so_dataset"


def save_vecotrizer(path, vectorizer):
    with open(path, 'wb') as fin:
        pickle.dump(vectorizer, fin)


def read_vecotrizer(path):
    with open(path, 'rb') as fin:
        vecotrizer = pickle.load(fin)
    return vecotrizer


def calculate_class_weights(labels, version='sklearn'):
    if version == 'sklearn':
        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    elif version == 'probs':
        class_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_count
        return weight
    else:
        # https://forums.fast.ai/t/correcting-class-imbalance-for-nlp/22152/6
        counts = Counter(labels)
        trn_weights = [count / len(labels) for idx, count in counts.items()]
        class_weights = np.array([max(trn_weights) / value for value in trn_weights])
    return class_weights  # make weights out of inverse counts
