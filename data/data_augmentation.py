#!/usr/bin/env python

"""
Implements the data augmentation pipeline for this project
"""

### IMPORTS ###
# Built-in imports
from typing import Tuple

# Lib imports
from torchvision import transforms
from torchvision.transforms import v2

import torch

# Custom imports


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### STATIC FUNCTIONS ###
def create_data_augmentation(image_size: Tuple[int, int], flip_image: bool = False, rotate_image: bool = False, 
                             standardize: bool = False, standardize_mean: torch.tensor = torch.tensor([125, 125, 125]), 
                             standardize_std: torch.tensor = torch.tensor([50, 50, 50])) -> transforms.Compose:
    """
    This function creates a data augmentation pipeline for the training of the model

    Parameters:
    ----------
    image_size: Tuple[int, int]
        The size of the image to be used for the training
    flip_image: bool
        Whether to flip the image or not
    rotate_image: bool
        Whether to rotate the image or not
    standardize: bool, optional
        Whether to standardize the image or not
        Default: False
    standardize_mean: torch.Tensor, optional
        The mean value to use for standardization
        Default: torch.tensor([125, 125, 125])
    standardize_std: torch.Tensor, optional
        The standard deviation to use for standardization
        Default: torch.tensor([50, 50, 50])

    Returns:
    ----------
    transforms.Compose
        The data augmentation pipeline
    """

    # Create the list of augmentations
    augmentation_list = []

    # Data standardization
    if standardize:
        augmentation_list.append(transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]))

    # Add flip augmentation
    if flip_image:
        augmentation_list.append(transforms.RandomHorizontalFlip())
        augmentation_list.append(transforms.RandomVerticalFlip())

    # Add rotation augmentation
    if rotate_image:
        augmentation_list.append(transforms.RandomRotation(degrees=(0, 180)))        

    # Add resizing augmentation
    augmentation_list.append(transforms.Resize((image_size[0], image_size[1]), antialias=True))

    return transforms.Compose(augmentation_list)
