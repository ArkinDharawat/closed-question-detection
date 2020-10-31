#!/usr/bin/env python
# coding: utf-8
import argparse
import os

import pandas as pd
from spacy.lang.en import English

from data_extraction_cleaning.utils import body_strip_tags, just_text, filter_sentence, get_tag_list, remove_filpaths, TOKEN_SEP

FOLDER_PATH = "../so_dataset"
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def tokenize(text, remove_stop_words=False, use_lemma=False):
    tokens = []
    doc = tokenizer(text)
    for token in doc:
        if 'd' not in token.shape_ and token.is_alpha:
            if remove_stop_words and token.is_stop:
                continue # ignore stop words
            # print(token.is_stop)
            if use_lemma:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)
    return TOKEN_SEP.join(tokens)


def main():
    parser = argparse.ArgumentParser(description='Train random forrest model')
    parser.add_argument('--output', type=str, help='output path')
    parser.add_argument('--lemma', type=eval, choices=[True, False], default='False',
                        help='lemmatize the tokens ')
    parser.add_argument('--stop', type=eval, choices=[True, False], default='False', help='remove stop words')
    args = parser.parse_args()

    use_lemma = args.lemma
    rem_stop_words = args.stop

    df_total_features = pd.read_csv(os.path.join(FOLDER_PATH, "so_questions_labelled.csv"))

    # filter lowercase titles
    df_total_features['title'] = df_total_features['Title'].apply(lambda x: ' '.join(filter_sentence(x)).lower())
    df_total_features['tag_list'] = df_total_features['Tags'].apply(lambda x: get_tag_list(x))  # get list of tags

    df_total_features['body'] = df_total_features['Body'].apply(just_text).apply(body_strip_tags)  # clean out tags
    df_total_features['body'] = df_total_features['body'].apply(remove_filpaths)  # remove filepaths

    df_cleaned = df_total_features.dropna(how='any')  # clear nan
    print(f'size after clearning nans {df_cleaned.shape}')

    # tokenize body and title
    def tokenize_text(x):
        return tokenize(x, use_lemma=use_lemma, remove_stop_words=rem_stop_words)
    df_cleaned['title'] = df_cleaned['title'].apply(tokenize_text)
    df_cleaned['body'] = df_cleaned['body'].apply(tokenize_text)
    # re-name columns
    df_cleaned['qid'] = df_cleaned['Qid']
    df_cleaned['label'] = df_cleaned['Label']

    # write only feature columns
    feature_columns = ['qid', 'title', 'body', 'tag_list', 'label']
    df_cleaned = df_cleaned[feature_columns]  # get only featured columns
    df_cleaned = df_cleaned.replace(r'^\s*$', pd.NA, regex=True)  # empty string as NaNs
    df_cleaned = df_cleaned.dropna(how='any')  # clear nan
    df_cleaned.to_csv(os.path.join(FOLDER_PATH, args.output), index=False)


if __name__ == '__main__':
    main()
