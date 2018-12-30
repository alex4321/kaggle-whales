import os
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import common
import config


def _load_image(fname):
    image = imread(fname)
    image = resize(image,
                   config.IMAGES_SIZE,
                   mode='constant',
                   anti_aliasing=True,
                   preserve_range=True)
    if list(image.shape) != list(config.IMAGES_SIZE_WITH_CHANNELS):
        image = np.stack([image, image, image], axis=-1).reshape(config.IMAGES_SIZE_WITH_CHANNELS)
    image = image.astype(np.uint8)
    return image


def _load_directory(directory):
    files = os.listdir(directory)
    images = Parallel(n_jobs=config.IMAGE_PREPARE_JOBS) (
        delayed(_load_image)(os.path.join(directory, fname))
        for fname in tqdm(files)
    )
    return dict(zip(files, images))


if __name__ == '__main__':
    common.write_pickle(
        _load_directory(config.TRAIN_DIR),
        config.TRAIN_IMAGES_PICKLE
    )
    common.write_pickle(
        _load_directory(config.TEST_DIR),
        config.TEST_IMAGES_PICKLE
    )
