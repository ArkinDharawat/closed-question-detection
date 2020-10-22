import pandas as pd
import os

FOLDER_PATH = "../so_dataset"

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(FOLDER_PATH, "so_questions_labelled.csv"))
    print(df['Label'].value_counts())