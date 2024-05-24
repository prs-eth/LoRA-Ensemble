#!/usr/bin/env python

"""
Implements the training pipeline for this project
"""

### IMPORTS ###
# Built-in imports

# Lib imports
import torch

# Custom imports


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### FUNCTIONS ###
def set_gpu() -> torch.device:
    """
    Set the device to use for training and inference

    Returns
    -------
    device : torch.device
        The device to use for training and inference
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training and inference")
    
    return device


def statistics_gpu_memory() -> None:
    """
    Print the statistics of the GPU memory.
    """
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t - r -a   # free inside reserved
    
    # Transform bytes to gigabytes
    t_gb = t / (1024**3)
    r_gb = r / (1024**3)
    a_gb = a / (1024**3)
    f_gb = f / (1024**3)
    
    print(f"Total memory: {t_gb:.2f} GB, Reserved memory: {r_gb:.2f} GB, Allocated memory: {a_gb:.2f} GB, "
          f"Free memory: {f_gb:.2f} GB")


# Device singleton
DEVICE = set_gpu()
