import pickle


def read_pickle(path):
    with open(path, 'rb') as src:
        return pickle.load(src)


def write_pickle(data, path):
    with open(path, 'wb') as target:
        pickle.dump(data, target)
