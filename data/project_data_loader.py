#!/usr/bin/env python

"""
Implements the data loaders for this project
"""

### IMPORTS ###
# Built-in imports
import pickle

# Lib imports
import torch 
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom imports


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### CLASS DEFINITION ###
class HAM10000(Dataset):
    """
    A dataloader for the HAM10000 dataset.
    """
     
    def __init__(self, labels, dataset_path, transform=None):
        """
        A dataloader for the HAM10000 dataset.

        Parameters
        ----------
        labels : pd.DataFrame
            The dataframe containing the labels
        dataset_path : Path
            The path to the dataset
        transform : torchvision.transforms
            Transformation to apply to the images

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label

        Labels in the dataset:
        ----------------------
        - Benign keratosis-like lesions (bkl): 0 (∼11.0%)
        - Melanocytic nevi (nv): 1 (∼66.9%)
        - Dermatofibroma (df): 2 (∼1.1%)
        - Melanoma (mel):	3 (∼11.1%)
        - Vascular lesions (vasc): 4 (∼1.4%)
        - Basal cell carcinoma (bcc): 5 (∼5.1%)
        - Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec): 6 (∼3.2%)
        """

        self.labels = labels
        self.path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # get image name
        img_name = self.labels.iloc[idx, 0]

        # get image path
        img_path = self.path.joinpath('HA_10000_images').joinpath(img_name + '.jpg')

        # get image
        image = plt.imread(img_path)

        # get label
        label = self.labels.iloc[idx, 2]

        # convert image to tensor and permute the dimensions
        image = torch.tensor(image).permute(2, 0, 1).float()

        # apply transformation
        if self.transform:
            image = self.transform(image)

        return image, label
    

def unpickle(file):
    """
    Function to unpickle a file

    Parameters
    ----------
    file : str
        Path to the file to unpickle

    Returns
    -------
    dict
        The unpickled dictionary
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10(Dataset):
    """
    A dataloader for the CIFAR10 dataset.
    """

    def __init__(self, batch_files, transform=None):
        """
        A dataloader for the CIFAR10 dataset.

        Parameters
        ----------
        batch_files : List
            List of paths to the batch files
        transform : torchvision.transforms
            Transformation to apply to the images

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        # Create a list to store the data and labels
        self.data = []
        self.labels = []

        # Load the data from the batch files
        for file in batch_files:
            # Unpickle the file
            batch_data = unpickle(file)
            # Append the data and labels to the lists
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'labels'])
        # Create a tensor from the data
        self.data = np.concatenate(self.data, axis=0)  
        self.data = torch.tensor(self.data)
        # Create a tensor from the labels
        self.labels = torch.tensor(self.labels)
        # Store the transformation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters
        ----------
        idx : int
            The index of the image to return

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        img = self.data[idx]
        img = img.reshape(3, 32, 32)  
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img.float())
        return img, label


class CIFAR100(Dataset):
    """
    A dataloader for the CIFAR100 dataset.
    """

    def __init__(self, batch_files, transform=None):
        """
        A dataloader for the CIFAR100 dataset.

        Parameters
        ----------
        batch_files : list
            List of paths to the batch files
        transform : torchvision.transforms
            Transformation to apply to the images

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        # Create a list to store the data and labels
        self.data = []
        self.labels = []

        # Load the data from the batch files
        for file in batch_files:
            # Unpickle the file
            batch_data = unpickle(file)
            # Append the data and labels to the lists
            self.data.append(batch_data[b'data'])
            self.labels.extend(batch_data[b'fine_labels'])
        # Create a tensor from the data
        self.data = np.concatenate(self.data, axis=0)
        self.data = torch.tensor(self.data)
        # Create a tensor from the labels
        self.labels = torch.tensor(self.labels)
        # Store the transformation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters
        ----------
        idx : int
            The index of the image to return

        Returns
        -------
        img : torch.Tensor
            The image
        label : torch.Tensor
            The label
        """

        img = self.data[idx]
        img = img.reshape(3, 32, 32)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img.float())
        return img, label 
