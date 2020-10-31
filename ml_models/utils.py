import pickle
FOLDER_PATH = "so_dataset"

def save_vecotrizer(path, vectorizer):
    with open(path, 'wb') as fin:
        pickle.dump(vectorizer, fin)


def read_vecotrizer(path):
    with open(path, 'rb') as fin:
        vecotrizer = pickle.load(fin)
    return vecotrizer
