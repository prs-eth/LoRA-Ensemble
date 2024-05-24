#!/usr/bin/env python

"""
Implements the learning rate schedule for this project
"""

### IMPORTS ###
# Built-in imports
import math

# Lib imports
import torch

# Custom imports
from torch.optim.lr_scheduler import LambdaLR


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### FUNCTIONS ###
class Cosine_Schedule(LambdaLR):
    """
    Linear warmup followed by a cosine decline in learning rate.
    Initially, the learning rate ramps up linearly from 0 to 1 across a specified number of warm-up steps during training.
    Subsequently, it follows a cosine pattern for the remaining steps (total steps minus warm-up steps).
    """

    def __init__(self, optimizer: torch.optim, steps_warmup: int, steps_total: int, cycles: float =.5, last_epoch: int = -1):
        """
        Linear warmup followed by a cosine decline in learning rate.
        Initially, the learning rate ramps up linearly from 0 to 1 across a specified number of warm-up steps during training.
        Subsequently, it follows a cosine pattern for the remaining steps (total steps minus warm-up steps).

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which the learning rate is scheduled.
        steps_warmup : int
            Number of warm-up steps.
        steps_total : int
            Total number of training steps.
        cycles : float, optional
            Number of cosine cycles (default is 0.5).
        last_epoch : int, optional
            The index of the last epoch (default is -1).
        """

        self.steps_warmup = steps_warmup
        self.steps_total = steps_total
        self.cycles = cycles
        super(Cosine_Schedule, self).__init__(optimizer, self.lr_lambda, last_epoch = last_epoch)

    def lr_lambda(self, step: int) -> float:
        """
        Compute the learning rate at a given step.

        Parameters
        ----------
        step : int
            Training step.

        Returns
        -------
        float
            Learning rate at the given step.
        """

        # warmup
        if step < self.steps_warmup:
            return float(step) / float(max(1.0, self.steps_warmup))
        
        # follow cosine pattern after warmup
        progress = float(step - self.steps_warmup) / float(max(1, self.steps_total - self.steps_warmup))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class Constant_Schedule(LambdaLR):
    """
    Linear warmup followed by a constant learning rate.
    Initially, the learning rate ramps up linearly from 0 to 1 across a specified number of warm-up steps during training.
    Subsequently, it remains constant for the remaining steps.
    """


    def __init__(self, optimizer: torch.optim, steps_warmup: int, last_epoch: int = -1):
        """
        Linear warmup followed by a constant learning rate.
        Initially, the learning rate ramps up linearly from 0 to 1 across a specified number of warm-up steps during training.
        Subsequently, it remains constant for the remaining steps.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which the learning rate is scheduled.
        steps_warmup : int
            Number of warm-up steps.

        returns
        -------
        float
            Learning rate at the given step.
        """

        self.steps_warmup = steps_warmup
        super(Constant_Schedule, self).__init__(optimizer, self.lr_lambda, last_epoch = last_epoch)

    def lr_lambda(self, step):
        if step < self.steps_warmup:
            return float(step) / float(max(1.0, self.steps_warmup))
        return 1.


class Linear_Schedule(LambdaLR):
    """ 
    Linear warmup followed by a linear decline in learning rate.
    """

    def __init__(self, optimizer: torch.optim, steps_warmup: int, steps_total: int, last_epoch: int = -1):
        """
        Linear warmup followed by a linear decline in learning rate.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which the learning rate is scheduled.
        steps_warmup : int
            Number of warm-up steps.
        steps_total : int
            Total number of training steps.

        returns
        -------
        float
            Learning rate at the given step.
        """

        self.steps_warmup = steps_warmup
        self.steps_total = steps_total
        super(Linear_Schedule, self).__init__(optimizer, self.lr_lambda, last_epoch = last_epoch)

    def lr_lambda(self, step):
        if step < self.steps_warmup:
            return float(step) / float(max(1, self.steps_warmup))
        return max(0.0, float(self.steps_total - step) / float(max(1.0, self.steps_total - self.steps_warmup)))
