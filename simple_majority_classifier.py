import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

from model_metrics import get_metrics


def train_majority_classifier():
    # set seed and any other hyper-parameters
    random_seed = 42
    train_test_split_ratio = 0.2

    # read data
    FOLDER_PATH = "so_dataset"
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
    q_bodies = df['body'].apply(lambda x: x.split('|'))
    q_titles = df['title'].apply(lambda x: x.split('|'))
    q_tags = df['tag_list'].apply(lambda x: x.split('|'))
    labels = df['label']

    # assign features
    X = q_titles.index
    y = df['label']

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio,
                                                        random_state=random_seed)

    # train model
    majority_label = np.argmax(np.bincount(y_train))

    # get predicted labels
    y_pred_test = np.ones(y_test.shape) * majority_label

    # generate metrics
    get_metrics(y_pred=y_pred_test, y_true=y_test, save_dir="./", model_name='majority_classifier')


if __name__ == '__main__':
    train_majority_classifier()
