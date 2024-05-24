#!/usr/bin/env python

"""
Functions for data loading and processing
"""

### IMPORTS ###
# Built-in imports
from typing import Tuple

# Lib imports
import torch 
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Custom imports


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"

def class_weights_inverse_num_of_samples(nr_classes, samples_per_class, power=1):
    """
    Function to calculate class weights based on the inverse number of samples per class. 
    The power parameter can be used to adjust the weights. Popular values are 0.5 (square root) or 1.
    The weights are normalized to sum to 1.

    Source
    ------
    https://medium.com/gumgum-tech/handling-class-imbalance-by-introducing-sample-weighting-in-the-loss-function-3bdebd8203b4

    Parameters
    ----------
    nr_classes : int
        The number of classes
    samples_per_class : list
        The number of samples per class
    power : int, optional
        The power to apply to the weights, by default 1
        
    Returns
    -------
    class_weights : list
        The class weights
    """

    # calculate class weights
    class_weights = 1.0 / np.array(np.power(samples_per_class, power))
    class_weights = class_weights / np.sum(class_weights) * nr_classes

    # normalize class weights
    class_weights = class_weights / class_weights.sum()
    return class_weights


def class_weights_effective_num_of_samples(nr_classes, samples_per_class, beta):
    """
    Function to calculate class weights based on the effective number of samples per class.
    The beta parameter can be used to adjust the weights. A beta value of 0 corresponds to uniform class weights,
    while a beta value near 1 (e.g., 0.9999) corresponds to class weights based on the inverse number of samples.
    Popular values are 0.9, 0.99, 0.999, 0.9999. The weights are normalized to sum to 1.

    Source
    ------
    Cui, Y., Jia, M., Lin, T., Song, Y., & Belongie, S.J. (2019). Class-Balanced Loss Based on Effective Number of Samples. 
    2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9260-9269.

    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf

    Parameters
    ----------
    nr_classes : int
        The number of classes
    samples_per_class : list
        The number of samples per class
    beta : float
        The beta parameter

    returns
    -------
    class_weights : list
        The class weights
    """

    # calculate class weights
    effective_num = 1.0 - np.power(beta, samples_per_class)
    class_weights = (1.0 - beta) / np.array(effective_num)
    class_weights = class_weights / np.sum(class_weights) * nr_classes

    # normalize class weights
    class_weights = class_weights / class_weights.sum()
    return class_weights


def download_CIFAR10(path: str, train: bool = True, download: bool = True, transform: bool = None) -> None:
    """
    Function for downloading CIFAR10 dataset.

    Parameters
    ----------
    path : str
        Path to store the dataset
    train : bool, optional
        Whether to download the training set, otherwise test set, by default True
    download : bool, optional
        Whether to download the dataset, by default True
    transform : torchvision.transforms, optional
        Transformation to apply to the images, by default None
    """

    torchvision.datasets.CIFAR10(root=path, train=train, download=download, transform=transform)


def download_CIFAR100(path: str, train: bool = True, download: bool = True, transform: bool = None) -> None:
    """
    Function for downloading CIFAR100 dataset.

    Parameters
    ----------
    path : str
        Path to store the dataset
    train : bool, optional
        Whether to download the training set, otherwise test set, by default True
    download : bool, optional
        Whether to download the dataset, by default True
    transform : torchvision.transforms, optional
        Transformation to apply to the images, by default None
    """

    torchvision.datasets.CIFAR100(root=path, train=train, download=download, transform=transform)


def calculate_pixel_mean_and_variance(data_loader: DataLoader, image_with: int = 32, image_height: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:   
    """
    Function to calculate the mean and variance of the pixel values in the dataset

    Parameters
    ----------
    data_loader : DataLoader
        The data loader for the dataset
    image_with : int, optional
        The width of the images, by default 32
    image_height : int, optional

    Returns
    -------
    channel_mean : torch.Tensor
        The mean of the pixel values per channel
    channel_variance : torch.Tensor
        The variance of the pixel values per channel
    """

    # initialize variables
    num_samples = 0
    red_sum = 0
    green_sum = 0
    blue_sum = 0
    red_squared_diff = 0
    green_squared_diff = 0
    blue_squared_diff = 0
    
    # iterate over the dataset
    for images, _ in data_loader:
        # get batch size
        batch_size = images.size(0)

        # update number of samples
        num_samples += batch_size

        # compute sum of pixel values per channel
        red_sum += torch.sum(images[:,0, :,:])
        green_sum += torch.sum(images[:,1, :,:])
        blue_sum += torch.sum(images[:,2, :,:])

    # compute mean
    red_mean = (red_sum / (num_samples * image_with * image_height)).float()
    green_mean = (green_sum / (num_samples * image_with * image_height)).float()
    blue_mean = (blue_sum / (num_samples * image_with * image_height)).float()

    # iterate over the dataset again
    for images, _ in data_loader:
        
        # compute squared difference for std calculation
        red_squared_diff += torch.sum((torch.sub(images[:,0, :,:].flatten(), red_mean)) ** 2)
        green_squared_diff += torch.sum((torch.sub(images[:,1, :,:].flatten(), green_mean)) ** 2)
        blue_squared_diff += torch.sum((torch.sub(images[:,2, :,:].flatten(), blue_mean)) ** 2)

    # compute std
    red_std = torch.sqrt(red_squared_diff / (num_samples * image_with * image_height)).item()
    green_std = torch.sqrt(green_squared_diff / (num_samples * image_with * image_height)).item()
    blue_std = torch.sqrt(blue_squared_diff / (num_samples * image_with * image_height)).item()

    # return mean and variance
    channel_mean = torch.tensor([red_mean, green_mean, blue_mean])
    channel_std = torch.tensor([red_std, green_std, blue_std])
    return channel_mean, channel_std


def denormalize_visualize_images(dataloader: DataLoader, mean: torch.tensor, std: torch.tensor, class_names: dict, num_images: int=10) -> None:
    """
    Visualize images from a dataset.

    Parameters:
    ----------
    dataloader : DataLoader
        The data loader for the dataset
    mean : torch.Tensor
        The mean of the pixel values per channel
    std : torch.Tensor
        The standard deviation of the pixel values per channel
    class_names : dict
        The class names
    num_images : int, optional
        The number of images to visualize, by default 10
    """

    count = 0
    for images, labels in dataloader:
        # denormalize images
        image = images[0].permute(1, 2, 0).numpy()  # Assuming images[0] is a tensor
        for i in range(3):  # Assuming 3 channels (RGB)
            image[:, :, i] = (image[:, :, i] * std[i].item()) + mean[i].item()
        image = np.clip(image.astype(np.uint8), 0, 255)
        plt.imshow(image)
        plt.title(class_names[labels[0].item()])
        plt.show()
        count += 1
        if count == num_images:
            break
