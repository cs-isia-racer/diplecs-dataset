import os
import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset
from skimage import io, color
from torchvision import transforms
import numpy as np


def get_paths():
    """returns the paths for the project
    :returns: (img_path_root, images, dataset_files)
    """

    img_path = "224_dataset/images"

    images = [
        os.path.join(img_path, p) for p in os.listdir(img_path) if p.endswith(".jpg")
    ]
    dat_files = [
        os.path.join("224_dataset/control", p)
        for p in os.listdir("224_dataset/control/")
        if p.endswith(".dat")
    ]

    return (img_path, images, dat_files)


def normalize_df(df):
    """normalizes the given dataframe
    TODO: better normalization (a reproducible one)

    :df: A dataframe containing at least the throttle and steering columns
    :returns: a normalized dataframe (values in [0, 1] for the throttle and steering columns)

    """
    max_steering, min_steering = df["steering"].max(), df["steering"].min()
    max_throttle, min_throttle = df["throttle"].max(), df["throttle"].min()

    df["steering"] = (df["steering"] - min_steering) / (max_steering - min_steering)
    df["throttle"] = (df["throttle"] - min_throttle) / (max_throttle - min_throttle)
    return df


def read_dataset(dataset_files, normalize=True):
    """reads the dataset and returns a dataframe

    :dataset_files: a list of the paths to the dataset_files
    :normalize: whether we should normalize the dataframe or not (defaults to True)
    :returns: a dataframe with 3 columns (image_path, throttle, steering)

    """
    frames = []

    for dat_file in dataset_files:
        df = pd.read_csv(dat_file, sep="\t", names=["image", "throttle", "steering"])
        prefix = os.path.splitext(os.path.basename(dat_file))[0]

        df["image"] = prefix + df["image"]
        # Drop the last 2 rows (for some reason they are not in the dataset)
        df.drop(df.tail(2).index, inplace=True)

        frames.append(df)

    frames = pd.concat(frames)

    if normalize:
        frames = normalize_df(frames)

    return frames


# This will implement the Dataset class for our Image Dataset
class ImageDataset(Dataset):
    def __init__(self, dataset_df, root_dir, limit=None, transform=None):
        self.df = dataset_df

        if limit is not None:
            self.df = self.df.iloc[:limit]

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])

        # convert into RGB to suppors transfer learning using the Resnet
        image = color.gray2rgb(io.imread(img_name))
        feats = self.df.iloc[idx, 1:]
        feats = np.array([feats]).astype("float").reshape(-1, 2)

        sample = {"image": image, "feats": feats}

        if self.transform:
            sample = self.transform(sample)

        return sample


# From: https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua#L67
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def to_tensor(sample):
    """transforms the given sample to a tensor

    :sample: a dict {image, feats}
    :returns: a dict {image, feats} in tensor format

    """
    image, feats = sample["image"], sample["feats"]

    # From: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))

    # Convert to Floats
    return {
        "image": normalize(torch.from_numpy(image).type("torch.FloatTensor")),
        "feats": torch.from_numpy(feats).type("torch.FloatTensor").view(2),
    }
