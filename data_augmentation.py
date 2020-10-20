import pandas as pd
import os
import numpy as np
import nlpaug.augmenter.word as naw
from joblib import Parallel, delayed

FOLDER_PATH = "so_dataset"
"""
Download w2v embs
- pip install gdown
- gdown --id  0B7XkCwpI5KDYNlNUTTlSS21pQmM
"""
MODEL_DIR = "./"  # path to google. glove and other embs models
DEL_SINGLE = True
USE_W2V = False


def get_augmentations(max_words_to_augment=2, aug_percentage=0.01, w2v_top_k=5):
    word2vec_model_path = os.path.join(MODEL_DIR, 'GoogleNews-vectors-negative300.bin')

    augmentations = []
    aug_random_swap = naw.RandomWordAug(action="swap", aug_max=max_words_to_augment, aug_p=aug_percentage)
    augmentations.append(aug_random_swap)
    if DEL_SINGLE:
        # delete only a single token
        aug_delete_swap = naw.RandomWordAug(action="delete", aug_max=1, aug_p=aug_percentage)
    else:
        aug_delete_swap = naw.RandomWordAug(action="delete", aug_max=max_words_to_augment, aug_p=aug_percentage)
    augmentations.append(aug_delete_swap)

    if USE_W2V:
        aug_w2v = naw.WordEmbsAug(model_type='word2vec', model_path=word2vec_model_path, action="substitute",
                                  top_k=w2v_top_k)
        augmentations.append(aug_w2v)

    return augmentations


def augment_text(text, num_aug, aug_prob, max_words):
    augmented_text = text.replace('|', ' ')
    augmentations_list = get_augmentations(max_words_to_augment=max_words, aug_percentage=aug_prob)
    augmentations = np.random.choice(augmentations_list, replace=True, size=num_aug)
    for aug in augmentations:
        augmented_text = aug.augment(augmented_text, n=1, num_thread=1)
    return augmented_text.replace(' ', '|')


def main():
    global DEL_SINGLE
    # TODO: make these parse augs
    augmented_dataset = os.path.join(FOLDER_PATH, "so_questions_augmented.csv")
    max_words = 3
    aug_prob = 0.1
    num_aug = 2

    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))

    pos_index = df[df['label'] > 0].index  # postive labels

    q_titles = df['title'].iloc[pos_index]
    q_bodies = df['body'].iloc[pos_index]

    augmented_titles = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(augment_text)(title, num_aug, aug_prob, max_words) for title in q_titles)

    DEL_SINGLE = False  # delete multiple tokens
    max_words = 10  # increase maximum words to augment
    augmented_bodies = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(augment_text)(title, num_aug, aug_prob, max_words) for title in q_bodies)

    aug_tags = df['tag_list'].iloc[pos_index]
    aug_labels = df['label'].iloc[pos_index]

    data_dict = {
        'title': augmented_titles,
        'body': augmented_bodies,
        'tag_list': aug_tags,
        'label': aug_labels
    }
    df_augmented = pd.DataFrame.from_records(data_dict)

    # debug
    import code
    code.interact(local={**locals(), **globals()})

    df_new = df.append(df_augmented)
    df_new['qid'] = df_new['qid'].fillna(0)  # all augmented quids are 0
    df_new.to_csv(augmented_dataset, index=False)


if __name__ == '__main__':
    main()
