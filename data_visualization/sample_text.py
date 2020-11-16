import os
from collections import Counter

import pandas as pd

FOLDER_PATH = "so_dataset"

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(FOLDER_PATH, "so_questions_labelled.csv"))
    label_types = df['label'].unique()
    sample_size = 3

    for label in label_types:
        print(label)
        indices = df[df['label'] == label].index
        titles = df['title'].iloc[indices]
        # bodies = df['body'].iloc[indices]
        tags = df['tag_list'].iloc[indices]

        print("Some Sample titles:")
        for index in titles.sample(n=sample_size, random_state=10101).index:
            print(df.iloc[index]['Qid'])
        print("------------")
        tags = tags.apply(lambda x: x.replace('<', '')[:-1].split('>'))
        all_tags = [t for tag_list in tags for t in tag_list]
        c = Counter(all_tags)
        print(f"Top 5 tags: {c.most_common(5)}")
