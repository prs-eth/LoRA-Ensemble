#!/usr/bin/env python

"""
Implements utility functions for the training pipeline.
"""

### IMPORTS ###
# Built-in imports
import os
import random

# Lib imports
import numpy as np
import torch

# Custom imports
import const


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### FUNCTIONS ###
def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducability of results

    Parameters
    ----------
    seed : int
        Random seed number.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
