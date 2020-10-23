import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_models.tfidf_vectorize import build_tfidf_vectorizer
from model_metrics import get_metrics

FOLDER_PATH = "../so_dataset"


def train_model():
    # TODO: tune model
    random_seed = 42
    train_test_split_ratio = 0.8

    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))

    q_bodies = df['body'].apply(lambda x: x.replace('|', ' ').lower())
    q_titles = df['title'].apply(lambda x: x.replace('|', ' ').lower())
    q_tags = df['tag_list'].apply(lambda x: x.replace('|', ' ').lower())

    # load vectorizers
    title_vectorizer, body_vectorizer, tag_vectorizer = build_tfidf_vectorizer()  # TODO: load from file

    # features
    X_title = title_vectorizer.transform(q_titles).toarray()
    X_body = body_vectorizer.transform(q_bodies).toarray()
    X_tag = tag_vectorizer.transform(q_tags).toarray()
    X = np.concatenate((X_title, X_body, X_tag), axis=1)
    y = y = df['label'].values

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio,
                                                        random_state=random_seed)
    print(f"Train dataset {X_train.shape}, {y_train.shape}")
    print(f"Test dataset {X_test.shape}, {y_test.shape}")

    # train
    # TODO: set class-weights
    clf = RandomForestClassifier(max_depth=-1, random_state=random_seed, n_jobs=8, verbose=1)  # 8 proccerssor
    clf.fit(X_train, y_train)

    # test
    y_pred = clf.predict(X_test)

    # generate metrics in folder
    get_metrics(y_pred=y_pred, y_true=y_test, save_dir="./", model_name='random_forrest')


if __name__ == '__main__':
    train_model()
