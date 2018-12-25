import pickle


IMAGE_SIZE = (224, 224)
IMAGE_SIZE_WITH_CHANNELS = list(IMAGE_SIZE) + [3]


def pickle_write(fname, data):
    with open(fname, 'wb') as target:
        pickle.dump(data, target)
