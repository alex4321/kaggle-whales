import pickle
import numpy as np


IMAGE_SIZE = (224, 224)
IMAGE_SIZE_WITH_CHANNELS = list(IMAGE_SIZE) + [3]


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
