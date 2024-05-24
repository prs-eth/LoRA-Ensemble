#!/usr/bin/env python

"""
Implements the training pipeline for this project
"""

### IMPORTS ###
# Built-in imports

# Lib imports
import torch
from torch import nn

# Custom imports
import const
from utils_GPU import DEVICE


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### STATIC FUNCTIONS ###
def load_model(model_name: str, model: nn.Module) -> nn.Module:
    """
    Load a model from a given path

    Parameters
    ----------
    model_name : str
        Name of the model to be loaded
    model : nn.Module
        A model object to load the parameters into

    Returns
    -------
    model : nn.Module
        The loaded model
    """

    # Construct the file name
    file_name = model_name + ".pt"
    model_path = const.MODEL_STORAGE_DIR.joinpath(file_name)

    # Load the model parameters
    model_state_dict = torch.load(model_path, map_location=DEVICE)

    # Assign the model parameters to the model
    model.set_params(model_state_dict)

    return model
