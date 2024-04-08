##################################################
# Class to create a modified dataset object for sensor data also containing meta information.
##################################################
# Author: Lloyd Pellatt
# Email: lp349@sussex.ac.uk
##################################################

import os
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from dl_har_dataloader.dataloader_utils import paint
from .dataset_utils import sliding_window, normalize, standardize

__all__ = ["SensorDataset"]


class SensorDataset(Dataset):
    """
    A dataset class for multi-channel time-series data captured by wearable sensors.
    This class is slightly modified from the original implementation at:
     https://github.com/AdelaideAuto-IDLab/Attend-And-Discriminate
    """

    def __init__(
        self,
        dataset,
        window,
        stride,
        stride_test,
        path_processed,
        name=None,
        prefix=None,
        verbose=False,
        lazy_load=False,
        scaling="standardize",
        min_vals=None,
        max_vals=None,
        mean=None,
        std=None,
        aug_list=None,
        aug_prob=None,
        aug_params=None
    ):
        """F
        Initialize instance.
        :param dataset: str. Name of target dataset.
        :param window: int. Sliding window size in samples.
        :param stride: int. Step size of the sliding window for training and validation data.
        :param stride_test: int. Step size of the sliding window for testing data.
        :param path_processed: str. Path to directory containing processed training, validation and test data.
        :param prefix: str. Prefix for the filename of the processed data. Options 'train', 'val', or 'test'.
        :param verbose: bool. Whether to print detailed information about the dataset when initializing.
        :param name: str. What to call this dataset (i.e. train, test, val).
        :param lazy_load: bool. Whether to load the whole windowed data into memory or not.
        :param scaling: str. What type of preprocessing to apply to the data. Options 'normalize', 'standardize', or None.
        :param min_vals: numpy array. Minimum values for each sensor channel. Used for normalization.
        :param max_vals: numpy array. Maximum values for each sensor channel. Used for normalization.
        :param mean: numpy array. Mean values for each sensor channel. Used for standardization.
        :param std: numpy array. Standard deviation values for each sensor channel. Used for standardization.
        :param aug_list: list. List of augmentation functions to apply to the data.
        :param aug_prob: list. List of probabilities for each augmentation function.
        """

        self.dataset = dataset
        self.window = window
        self.stride = stride
        self.stride_test = stride_test
        self.path_processed = path_processed
        self.verbose = verbose
        self.name = name
        self.lazy_load = lazy_load
        self.scaling = scaling
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.mean = mean
        self.std = std

        self.aug_list = aug_list
        self.aug_prob = aug_prob
        self.aug_params = aug_params



        if name is None:
            self.name = 'No name specified'
        if prefix is None:
            self.prefix = 'No prefix specified'
            self.path_dataset = glob(os.path.join(path_processed, '*.npz'))
        elif isinstance(prefix, str):
            self.prefix = prefix
            self.path_dataset = glob(os.path.join(path_processed, f'{prefix}.npz'))
        elif isinstance(prefix, list):
            self.prefix = prefix
            self.path_dataset = []
            for prefix in prefix:
                self.path_dataset.extend(glob(os.path.join(path_processed, f'{prefix}*.npz')))

        self.data = np.concatenate([np.load(path, allow_pickle=True)['data'] for path in self.path_dataset])
        self.target = np.concatenate([np.load(path, allow_pickle=True)['target'] for path in self.path_dataset])

        # Scale the data
        # if scaling != "normalize" and (self.min_vals is not None or self.max_vals is not None):
        #     raise ValueError(f"min_vals and max_vals cannot be specified when scaling is {scaling}.")

        if self.mean is None:
            self.mean = np.mean(self.data, axis=0)
        if self.std is None:
            self.std = np.std(self.data, axis=0)
            self.std[self.std == 0] = 1
        if self.min_vals is None:
            self.min_vals = np.min(self.data, axis=0)
        if self.max_vals is None:
            self.max_vals = np.max(self.data, axis=0)

        if self.scaling == 'normalize':
            self.data = normalize(self.data, min_vals=self.min_vals, max_vals=self.max_vals, verbose=self.verbose)
        elif self.scaling == 'standardize':
            self.data = standardize(self.data, self.mean, self.std)
        elif self.scaling is None:
            pass
        else:
            raise ValueError(f'Unknown preprocessing scheme {self.scaling}.')

        if self.aug_list is not None and self.prefix not in ["val", "test"]:
            for aug, prob, params in zip(self.aug_list, self.aug_prob, self.aug_params):
                print(f"Data Augmentation: {aug} with probability {prob} and params {params}")

        # To save memory, generate the windowed data on the fly
        if lazy_load:
            # Pre-calculate the number of windows
            self.len = (self.data.shape[0] - self.window) // self.stride + 1
        else:
            self.data, self.target = sliding_window(self.data, self.target, self.window, self.stride)
            self.len = self.data.shape[0]
            assert self.data.shape[0] == self.target.shape[0]

        if name is None:
            print(
                paint(
                    f"Creating {self.dataset} HAR dataset of size {self.len} ..."
                )
            )
        else:
            print(
                paint(
                    f"Creating {self.dataset} {self.name} HAR dataset of size {self.len} ..."
                )
            )

        self.n_channels = self.data.shape[-1] - 1
        self.n_classes = np.unique(self.target).shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        if self.lazy_load:
            start = index * self.stride
            end = start + self.window
            data = self.data[start:end]
            target = [int(self._get_label(start, end))]
        else:
            data = self.data[index]
            target = [int(self.target[index])]

        if self.aug_list is not None and self.prefix not in ["val", "test"]:
            for aug, prob, params in zip(self.aug_list, self.aug_prob, self.aug_params):
                if np.random.rand() < prob:
                    if params is None:
                        data = aug(data)
                    else:
                        data = aug(data, **params)

        data = torch.FloatTensor(data)
        target = torch.LongTensor(target)
        idx = torch.from_numpy(np.array(index))
        return data, target, idx

    def _get_label(self, start, end, scheme='max'):
        if scheme == 'last':
            return self.target[end - 1]

        elif scheme == 'max':
            return np.argmax(np.bincount(self.target[start:end]))

        else:
            raise ValueError(f"Unknown scheme {scheme}.")

