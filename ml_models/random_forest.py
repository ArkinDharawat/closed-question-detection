import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ml_models.tfidf_vectorize import build_tfidf_vectorizer
from model_metrics import get_metrics

FOLDER_PATH = "../so_dataset"


def train_model():
    # TODO: tune model
    random_seed = 42
    train_test_split_ratio = 0.8
    hyperparam_tune = True

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
    # TODO: class weight strategy
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights[0] = class_weights[0]  # * 0.01  # for 0th class
    class_weights = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights {class_weights}")
    if hyperparam_tune:
        model = RandomForestClassifier()
        tuning_parameters = {
             'n_estimators': [100, 500, 1000, 2500],
             'random_state': [42, 12345, 10101], # seed as hyperparamerter?
             'class_weight': ['balanced', 'balanced_subsample'],
             'max_features': ['auto', 'sqrt']
        }
        clf = GridSearchCV(model,
                           tuning_parameters,
                           scoring='%s_macro' % "f1",
                           n_jobs=-1)
        clf.fit(X_train, y_train)
        means = clf.cv_results_['mean_test_score']
        print(f"Mean F1 macro {means}")

    else:
        clf = RandomForestClassifier(n_estimators=500,
                                     random_state=random_seed,
                                     n_jobs=-1,
                                     class_weight='balanced_subsample',
                                     verbose=1)
        clf.fit(X_train, y_train)

    # test
    y_pred = clf.predict(X_test)

    # generate metrics in folder
    print(classification_report(y_test, y_pred))
    get_metrics(y_pred=y_pred, y_true=y_test, save_dir="./", model_name='rf_tfidf_model')


if __name__ == '__main__':
    train_model()
