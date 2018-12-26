import pickle
import numpy as np


IMAGE_SIZE = (224, 224)
IMAGE_SIZE_WITH_CHANNELS = list(IMAGE_SIZE) + [3]
TRAIN_IMAGES_SOURCE_PATH = '../input/train'
TRAIN_IMAGES_PICKLE_PATH = '../data/train-images.pkl'
TEST_IMAGES_SOURCE_PATH = '../input/test'
TEST_IMAGES_PICKLE_PATH = '../data/test-images.pkl'
VECTOR_SIZE = 300
TRAIN_MAPPING = '../input/train.csv'
TRIPLET_BATCH_SIZE = 8
EMBEDDING_BATCH_SIZE = 32
DEVICE = 'cuda'
TRIPLET_MODELS_DIRECTORY = '../data/triplet-models-{time}'
TRIPLET_MAX_EPOCHS = 100
TRIPLET_EARLY_STOPPING_PATIENCE = 10


def pickle_read(fname):
    with open(fname, 'rb') as src:
        return pickle.load(src)


def pickle_write(fname, data):
    with open(fname, 'wb') as target:
        pickle.dump(data, target)


def torch_preprocess_image(image):
    image = image.astype(np.float32) / 255.0  # Convert from 0..255 range to 0..1 range
    means = [0.485, 0.456, 0.406]  # See https://pytorch.org/docs/stable/torchvision/models.html
    stds = [0.229, 0.224, 0.225]
    for chanel in range(3):
        image[:, :, chanel] = (image[:, :, chanel] - means[chanel]) / stds[chanel]
    image = np.transpose(image, [2, 0, 1])  # From width,height,channel to channel,width,height
    return image
