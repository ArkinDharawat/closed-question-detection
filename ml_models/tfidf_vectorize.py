import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

from ml_models.utils import save_vectorizer, FOLDER_PATH


def build_vectorizer(df, vectorizer):
    q_bodies = df['body'].apply(lambda x: x.replace('|', ' ').lower())
    q_titles = df['title'].apply(lambda x: x.replace('|', ' ').lower())
    q_tags = df['tag_list'].apply(lambda x: x.replace('|', ' ').lower())

    if vectorizer == 0:
        # title vectorizer
        title_vectorizer = TfidfVectorizer(
            min_df=0.001,
            sublinear_tf=True
        )
        title_vectorizer.fit(q_titles)
        save_vectorizer('title_vectorizer.pk', title_vectorizer)

        # body vectorizer
        body_vectorizer = TfidfVectorizer(
            min_df=0.001,
            sublinear_tf=True
        )
        body_vectorizer.fit(q_bodies)
        save_vectorizer('body_vectorizer.pk', body_vectorizer)

        # title vectorizer
        tag_vectorizer = TfidfVectorizer(
            min_df=0.01,
            sublinear_tf=True
        )  # get rid of 10% of tags
        tag_vectorizer.fit(q_tags)
        save_vectorizer('tag_vectorizer.pk', tag_vectorizer)

    else:
        title_vectorizer = HashingVectorizer(n_features=2000)
        title_vectorizer.fit(q_titles)
        save_vectorizer('title_vectorizer.pk', title_vectorizer)
        body_vectorizer = HashingVectorizer(n_features=2000)
        body_vectorizer.fit(q_bodies)
        save_vectorizer('body_vectorizer.pk', body_vectorizer)
        tag_vectorizer = HashingVectorizer(n_features=2000)
        tag_vectorizer.fit(q_tags)
        save_vectorizer('tag_vectorizer.pk', tag_vectorizer)

    return title_vectorizer, body_vectorizer, tag_vectorizer


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
    build_vectorizer(df, 0)
    build_vectorizer(df, 1)
