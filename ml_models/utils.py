import pickle
import numpy as np
from sklearn.utils import class_weight
from collections import Counter

FOLDER_PATH = "..\\so_dataset"
#FOLDER_PATH = "so_dataset"
#I changed it to ..\\so_dataset because from the ml_models directory, I need to go back one step, then look for so_dataset
#and take the so_questions_cleaned.csv file. Before, it didn't go back so it said file doesn't exist.


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
        class_weights = 1. / class_count
    else:
        # https://forums.fast.ai/t/correcting-class-imbalance-for-nlp/22152/6
        counts = Counter(labels)
        trn_weights = [count / len(labels) for idx, count in counts.items()]
        class_weights = np.array([max(trn_weights) / value for value in trn_weights])
    return class_weights  # make weights out of inverse counts
