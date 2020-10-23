import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_models.utils import save_vecotrizer

FOLDER_PATH = "../so_dataset"


def build_tfidf_vectorizer():
    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))

    q_bodies = df['body'].apply(lambda x: x.replace('|', ' ').lower())
    q_titles = df['title'].apply(lambda x: x.replace('|', ' ').lower())
    q_tags = df['tag_list'].apply(lambda x: x.replace('|', ' ').lower())

    # title vectorizer
    title_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    title_vectorizer.fit(q_titles)
    save_vecotrizer('title_vectorizer.pk', title_vectorizer)

    # body vectorizer
    body_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    body_vectorizer.fit(q_bodies)
    save_vecotrizer('body_vectorizer.pk', body_vectorizer)

    # title vectorizer
    tag_vecotrizer = TfidfVectorizer(max_features=100)  # top 500 tags
    tag_vecotrizer.fit(q_tags)
    save_vecotrizer('tag_vectorizer.pk', tag_vecotrizer)

    return title_vectorizer, body_vectorizer, tag_vecotrizer


if __name__ == '__main__':
    build_tfidf_vectorizer()
