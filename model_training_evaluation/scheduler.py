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


class Cyclic_Cosine_Schedule(LambdaLR):
    """
    Cyclic cosine learning rate schedule.
    Follows a cosine annealing pattern within each cycle, resetting to the maximum learning rate at the beginning
    of each cycle. Commonly used in Snapshot Ensembles.
    """

    def __init__(
            self,
            optimizer: torch.optim,
            epochs: int,
            steps_per_epoch: int,
            num_cycles: int,
            min_lr: float = None,
            burn_in_epochs: int = 0,
            steps_warmup: int = 0,
            cos_cycles: float = .5,
            last_epoch: int = -1
    ):
        """
        Cyclic cosine learning rate schedule.
        Follows a cosine annealing pattern within each cycle, resetting to the maximum learning rate at the beginning
        of each cycle. Commonly used in Snapshot Ensembles.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for which the learning rate is scheduled.
        epochs : int
            Total number of training epochs. The number of epochs must be a multiple of num_cycles.
        steps_per_epoch : int
            The number of samples in a single epoch.
        num_cycles : int
            The number of steps per Snapshot cycle.
        min_lr : float, optional
            Minimum learning rate (default is 1e-6).
        burn_in_epochs : int, optional
            Number of epochs for the first cycle (default is 0). If set to 0, the first cycle will be a normal cycle.
            Otherwise, there is a slower Cosine annealing schedule. If warumup_steps is set, this period will also
            include linear warmup.
        steps_warmup : int, optional
            Number of warm-up steps.
        cos_cycles : float, optional
            Number of cosine cycles (default is 0.5).
        last_epoch : int, optional
            The index of the last epoch (default is -1).
        """

        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.cos_cycles = cos_cycles
        self.num_cycles = num_cycles

        num_normal_cycles = num_cycles - 1 if burn_in_epochs != 0 else num_cycles

        divisible = False
        while not divisible:
            if burn_in_epochs < epochs // num_cycles:
                self.burn_in_epochs = epochs // num_cycles
            else:
                self.burn_in_epochs = burn_in_epochs

            if (epochs - self.burn_in_epochs) % num_normal_cycles == 0:
                divisible = True
            else:
                burn_in_epochs += 1

        self.steps_warmup = steps_warmup
        self.burn_in_steps = self.burn_in_epochs * steps_per_epoch
        self.steps_per_cycle = (steps_per_epoch * (epochs - self.burn_in_epochs)) // num_normal_cycles
        if min_lr is None:
            min_lr = 1e-6
        self.min_lr = min_lr
        super(Cyclic_Cosine_Schedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

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

        # follow the cosine pattern within each snapshot cycle
        if self.burn_in_epochs != 0:
            if step <= self.burn_in_steps:
                if step < self.steps_warmup:
                    return float(step) / float(max(1.0, self.steps_warmup))
                else:
                    # follow cosine pattern after warmup
                    progress = float(step - self.steps_warmup) / float(max(1, self.burn_in_steps - self.steps_warmup))
                    return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cos_cycles) * 2.0 * progress)))

        alpha = step % self.steps_per_cycle
        return max(self.min_lr,
                   0.5 * (1. + math.cos(math.pi * float(self.cos_cycles) * 2.0 * alpha / float(self.steps_per_cycle))))

    def check_cycle_state(self, step: int) -> bool:
        """
        Check if the current cycle has ended.

        Parameters
        ----------
        step : int
            Training step.

        Returns
        -------
        bool
            True if the current cycle has ended.
        """

        if self.burn_in_epochs != 0:
            if step < self.burn_in_steps:
                return False
            elif step == self.burn_in_steps:
                return True
            else:
                return (step - self.burn_in_steps) % self.steps_per_cycle == 0

    def get_cycle_count(self, step: int) -> int:
        """
        Get the current cycle count.

        Parameters
        ----------
        step : int
            Training step.

        Returns
        -------
        int
            Current cycle count.
        """

        if self.burn_in_epochs != 0:
            if step < self.burn_in_steps:
                return 0
            else:
                return (step - self.burn_in_steps) // self.steps_per_cycle
