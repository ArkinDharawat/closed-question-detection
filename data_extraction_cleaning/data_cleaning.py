#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
from spacy.lang.en import English

from utils import body_strip_tags, just_text, filter_sentence, get_tag_list, remove_filpaths

FOLDER_PATH = "../so_dataset"
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def tokenize(text):
    tokens = []
    doc = tokenizer(text)
    for token in doc:
        if ('d' in token.shape_):
            # token contains digit
            continue
        else:
            tokens.append(token.text)
    return '|'.join(tokens)


def main():
    df_total_features = pd.read_csv(os.path.join(FOLDER_PATH, "so_questions_labelled.csv"))

    # filter lowercase titles
    df_total_features['title'] = df_total_features['Title'].apply(lambda x: ' '.join(filter_sentence(x)).lower())
    df_total_features['tag_list'] = df_total_features['Tags'].apply(lambda x: get_tag_list(x))  # get list of tags

    df_total_features['body'] = df_total_features['Body'].apply(just_text).apply(body_strip_tags)  # clean out tags
    df_total_features['body'] = df_total_features['body'].apply(remove_filpaths)  # remove filepaths

    df_cleaned = df_total_features.dropna(how='any')

    # tokenize body and title
    df_cleaned['title'] = df_cleaned['title'].apply(tokenize)
    df_cleaned['body'] = df_cleaned['body'].apply(tokenize)
    df_cleaned['label'] = df_cleaned['Label']

    # write only feature columns
    feature_columns = ['Qid', 'title', 'body', 'tag_list', 'label']
    df_cleaned = df_cleaned[feature_columns]  # get only featured columns
    df_cleaned.to_csv(os.path.join(FOLDER_PATH, "so_questions_cleaned.csv"), index=False)


if __name__ == '__main__':
    main()
