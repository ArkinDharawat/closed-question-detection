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


def get_augmentations(max_words_to_augment=2, aug_percentage=0.01, w2v_top_k=5):
    word2vec_model_path = os.path.join(MODEL_DIR, 'GoogleNews-vectors-negative300.bin')
    aug_random_swap = naw.RandomWordAug(action="swap", aug_max=max_words_to_augment, aug_p=aug_percentage)
    aug_delete_swap = naw.RandomWordAug(action="delete", aug_max=max_words_to_augment, aug_p=aug_percentage)
    # aug_w2v = naw.WordEmbsAug(model_type='word2vec', model_path=word2vec_model_path, action="substitute", top_k=w2v_top_k)
    # return [aug_w2v, aug_random_swap, aug_delete_swap]
    # TODO: add word 2 vec later
    return [aug_random_swap, aug_delete_swap]


def augment_text(text_list, num_aug, aug_prob, max_words):
    augmented_text = []
    augmentations_list = get_augmentations(max_words_to_augment=max_words, aug_percentage=aug_prob)
    augmentations = np.random.choice(augmentations_list, replace=True, size=num_aug)
    for aug in augmentations:
        augmented_text.append(aug.augment(text_list, n=1, num_thread=1))
    return augmented_text


def main():
    # TODO: make these parse augs
    augmented_dataset = os.path.join(FOLDER_PATH, "so_questions_augmented.csv")
    max_words = 3
    aug_prob = 0.05
    num_aug = 3

    df = pd.read_csv(os.path.join(FOLDER_PATH, 'so_questions_cleaned.csv'))
    q_titles = df['title'].apply(lambda x: x.split('|')).iloc[0]

    def augment_single_text(title):
        return augment_text(title, num_aug, aug_prob, max_words)

    augmented_titles = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(augment_text)(title, num_aug, aug_prob, max_words) for title in q_titles)


    # debug
    import code
    code.interact(local={**locals(), **globals()})


if __name__ == '__main__':
    main()
