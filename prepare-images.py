import os
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import common

N_JOBS = 4


def _load_image(fname):
    image = imread(fname)
    image = resize(image, common.IMAGE_SIZE_WITH_CHANNELS,
                   mode='constant',
                   anti_aliasing=True,
                   preserve_range=False)
    image = image.astype(np.uint8)
    if list(image.shape) != common.IMAGE_SIZE_WITH_CHANNELS:  # Grayscale images have 1 channel
        image = np.stack([image, image, image],
                         axis=-1)
        image = image.reshape(common.IMAGE_SIZE_WITH_CHANNELS)
    return image


def _load_directory(dirname):
    fnames = sorted(os.listdir(dirname))
    files = [os.path.join(dirname, file)
             for file in fnames]
    with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        images = parallel(joblib.delayed(_load_image)(file)
                          for file in tqdm(files))
    result = dict(zip(fnames, images))
    return result


if __name__ == '__main__':
    train_images = _load_directory(common.TRAIN_IMAGES_SOURCE_PATH)
    common.pickle_write(common.TRAIN_IMAGES_PICKLE_PATH, train_images)
    test_images = _load_directory(common.TEST_IMAGES_SOURCE_PATH)
    common.pickle_write(common.TEST_IMAGES_PICKLE_PATH, test_images)
