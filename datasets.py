from albumentations import (
    CLAHE, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, OneOf, Compose
)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import common


RANDOM_STATE = 42
TRAIN_IDS_COUNT = 0.9


def get_mappings(fname):
    df = pd.read_csv(fname)
    new_whale_idx = df['Id'] == 'new_whale'
    new_whale_df = df.loc[new_whale_idx].reset_index(drop=True)
    df = df.loc[~new_whale_idx].reset_index(drop=True)
    ids = pd.Series(df['Id'].unique())
    ids_shuffled = ids.sample(len(ids), random_state=RANDOM_STATE)
    train_ids_count = len(ids_shuffled) * TRAIN_IDS_COUNT
    val_ids_count = len(ids_shuffled) - train_ids_count
    train_ids = set(ids_shuffled.head(train_ids_count))
    val_ids = set(ids_shuffled.tail(val_ids_count))
    train_df = df.loc[df['Id'].isin(train_ids)].reset_index(drop=True)
    val_df = df.loc[df['Id'].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df, new_whale_df


def triplet_df_by_mapping(df):
    anchor = df.sort_values(['Id', 'Image'], ascending=[True, True]).reset_index(drop=True).add_suffix('_anchor')
    positive = df.sort_values(['Id', 'Image'], ascending=[True, False]).reset_index(drop=True).add_suffix('_positive')
    negative = df.sort_values(['Id', 'Image'], ascending=[False, True]).reset_index(drop=True).add_suffix('_negative')
    triplet = anchor.join(positive).join(negative)
    triplet_valid_idx = (triplet['Image_positive'] != triplet['Image_negative']) &\
                        (triplet['Image_anchor'] != triplet['Image_negative'])
    triplet = triplet.loc[triplet_valid_idx].reset_index(drop=True)
    triplet_shuffled = triplet.sample(len(triplet), random_state=RANDOM_STATE).reset_index(drop=True)
    return triplet_shuffled


class TripletDataset(Dataset):
    def __init__(self, images, triplet):
        super(TripletDataset, self).__init__()
        self.images = images
        self.triplet = triplet
        self.augmenter = self._build_augmenter()

    def __len__(self):
        return len(self)

    def _build_augmenter(self, p=0.5):
        return Compose([
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)

    def _preprocess_image(self, image):
        augmented = self.augmenter(image=image)['image']
        return common.torch_preprocess_image(augmented)

    def __getitem__(self, idx):
        anchor_image = self.images[self.triplet['Image_anchor'].values[idx]]
        positive_image = self.images[self.triplet['Image_positive'].values[idx]]
        negative_image = self.images[self.triplet['Image_negative'].values[idx]]
        anchor_X = self._preprocess_image(anchor_image)
        positive_X = self._preprocess_image(positive_image)
        negative_X = self._preprocess_image(negative_image)
        X = [anchor_X, positive_X, negative_X]
        y = []  # We don't really need ground truth value -
                # we need only to minimize anchor-positive distance while maximizing anchor-negative
        return X, y

    def build_dataloader(self, batch_size, *args, **kwargs):
        return DataLoader(self, batch_size=batch_size, *args, **kwargs)


class FeatureExtractorDataset(Dataset):
    def __init__(self, images, mapping):
        super(FeatureExtractorDataset, self).__init__()
        self.images = images
        self.mapping = mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image = self.images[self.mapping['Image'].values[idx]]
        image = common.torch_preprocess_image(image)
        X = [image]
        y = []  # We don't really need ground truth value -
                #   this dataset used only to extract features for KNN
        return X, y

    def build_dataloader(self, batch_size, *args, **kwargs):
        return DataLoader(self, batch_size=batch_size, *args, **kwargs)
