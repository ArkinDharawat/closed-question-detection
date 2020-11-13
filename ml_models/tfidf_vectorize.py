import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

from ml_models.utils import save_vecotrizer, FOLDER_PATH


def build_vectorizer(df, vectorizer):
    # TODO - Aaron: Try out HashingVecotrizer
    # TODO - Aaron: Look at: https://scikit-learn.org/0.15/auto_examples/document_classification_20newsgroups.html
    q_bodies = df['body'].apply(lambda x: x.replace('|', ' ').lower())
    q_titles = df['title'].apply(lambda x: x.replace('|', ' ').lower())
    q_tags = df['tag_list'].apply(lambda x: x.replace('|', ' ').lower())

    if vectorizer == 0:
        # title vectorizer
        title_vectorizer = TfidfVectorizer(
            # max_df=0.4,
            # stop_words='english',
            min_df=0.001,
            sublinear_tf=True
        )
        title_vectorizer.fit(q_titles)
        save_vecotrizer('title_vectorizer.pk', title_vectorizer)

        # body vectorizer
        body_vectorizer = TfidfVectorizer(
            # max_features=5000,
            # max_df=0.4,
            # stop_words='english',
            min_df=0.001,
            sublinear_tf=True
        )
        body_vectorizer.fit(q_bodies)
        save_vecotrizer('body_vectorizer.pk', body_vectorizer)

        # title vectorizer
        tag_vecotrizer = TfidfVectorizer(
            # max_features=500,
            # max_df=0.5,
            min_df=0.01,
            sublinear_tf=True
        )  # get rid of 10% of tags
        tag_vecotrizer.fit(q_tags)
        save_vecotrizer('tag_vectorizer.pk', tag_vecotrizer)

    elif vectorizer == 1:
        title_vectorizer = HashingVectorizer()
        title_vectorizer.fit(q_titles)
        save_vecotrizer('title_vectorizer.pk', title_vectorizer)
        body_vectorizer = HashingVectorizer()
        body_vectorizer.fit(q_bodies)
        save_vecotrizer('body_vectorizer.pk', body_vectorizer)
        tag_vectorizer = HashingVectorizer()
        tag_vectorizer.fit(q_tags)
        save_vecotrizer('tag_vectorizer.pk', tag_vectorizer)

    return title_vectorizer, body_vectorizer, tag_vectorizer


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
    build_vectorizer(df, 0)
    build_vectorizer(df, 1)
