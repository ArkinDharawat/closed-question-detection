import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
    y = df['label'].values

    # import code
    # code.interact(local={**locals(), **globals()})

    # train-test split
    X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=train_test_split_ratio,
                                                        random_state=random_seed)

    print(f"Train dataset {X_train.shape}, {y_train.shape}")
    print(f"Test dataset {X_test.shape}, {y_test.shape}")

    # train
    # TODO: set k-fold CV
    # TODO: class weight strategy
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights[0] = class_weights[0] * 0.01  # for 0th class
    class_weights = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights {class_weights}")
    clf = RandomForestClassifier(n_estimators=2000,
                                 criterion='entropy',
                                 random_state=random_seed,
                                 n_jobs=-1,
                                 class_weight=class_weights,
                                 verbose=1)
    clf.fit(X_train, y_train)
    print(clf.classes_, clf.class_weight)

    # test
    y_pred = clf.predict(X_test)

    # generate metrics in folder
    get_metrics(y_pred=y_pred, y_true=y_test, save_dir="./", model_name='random_forrest')


if __name__ == '__main__':
    train_model()
