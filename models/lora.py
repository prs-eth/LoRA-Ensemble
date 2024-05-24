#!/usr/bin/env python

"""
Implements a Low Rank Adaptation module
"""

### IMPORTS ###
# Built-in imports
from typing import List, Dict
import enum

# Lib imports
from torch import nn, Tensor, vmap
from torch.func import stack_module_state, functional_call
import copy

# Custom imports
from utils_GPU import DEVICE


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### CLASS DEFINITION ###
class Init_Weight(enum.Enum):
    NORMAL = 0
    KAIMING_UNIFORM = 1
    XAVIER_UNIFORM = 2
    DEFAULT = NORMAL


class LoRA(nn.Module):
    def __init__(
            self,
            w: nn.Module,
            rank: int,
            dim: int,
            initialize: bool = True,
            init_type: Init_Weight = Init_Weight.DEFAULT,
            init_settings: dict = None,
            out_dim: int = None
    ) -> None:
        """
        Implements a Low Rank Adaptation module

        Parameters
        ----------
        w : nn.Module
            The original projection layer
        rank : int
            The rank of the Low Rank Adaptation module
        dim : List[int]
            The dimension of the weight matrix
        initialize : bool, optional
            Whether to initialize the weights, by default True
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        out_dim : int, optional
            The output dimension of the LoRA layer, by default None
        """

        super().__init__()

        # LoRA rank
        self.rank = rank

        # Original projection layer weights
        self.w = w

        if out_dim is None:
            out_dim = dim

        # LoRA matrices
        self.w_a = nn.Linear(dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

        # Initialize the weights if needed
        if initialize:
            self.initialize_weights(init_type=init_type, init_settings=init_settings)

    def forward(self, x):
        """
        Forward pass for the LoRA Layer

        Parameters
        ----------
        x : Tensor
            The input tensor

        Returns
        -------
        out : Tensor
            The output tensor
        """

        out = self.w(x) + self.w_b(self.w_a(x))

        return out

    def initialize_weights(self, init_type: Init_Weight = Init_Weight.DEFAULT, init_settings: dict = None) -> None:
        """
        Initialize the weights of the LoRA matrices

        Parameters
        ----------
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        if init_type == Init_Weight.NORMAL:
            # Set the mean and standard deviation
            if init_settings is None:
                mean = 0
                std = 0.02
            else:
                mean = init_settings["mean"]
                std = init_settings["std"]

            # Initialize all weights separately
            nn.init.normal_(self.w_a.weight, mean=mean, std=std)
            nn.init.zeros_(self.w_b.weight)

        elif init_type == Init_Weight.KAIMING_UNIFORM:
            # Set the a_squared parameter
            if init_settings is None:
                a_squared = 5
            else:
                a_squared = init_settings["a_squared"]

            # Initialize all weights separately
            from math import sqrt
            nn.init.kaiming_uniform_(self.w_a.weight, a=sqrt(a_squared))
            nn.init.zeros_(self.w_b.weight)

        elif init_type == Init_Weight.XAVIER_UNIFORM:
            # Set the a_squared parameter
            if init_settings is None:
                gain  = 1
            else:
                gain = init_settings["gain"]

            # Initialize all weights separately
            nn.init.xavier_uniform_(self.w_a.weight, gain=gain)
            nn.init.zeros_(self.w_b.weight)
        else:
            raise ValueError("Invalid initialization type")


class EnsembleLoRA(nn.Module):
    """
    Class to ensemble the LoRA layers

    Credit
    ------
    https://pytorch.org/tutorials/intermediate/ensembling.html
    """

    def __init__(
            self,
            w: nn.Module,
            rank: int,
            dim: int,
            n_members: int,
            initialize: bool = True,
            init_type: Init_Weight = Init_Weight.DEFAULT,
            init_settings: dict = None,
            out_dim: int = None,
            chunk_size: int = None
    ):
        """
        Class to ensemble the LoRA layers

        Parameters
        ----------
        w : nn.Module
            The original projection layer
        rank : int
            The rank of the Low Rank Adaptation module
        dim : List[int]
            The dimension of the weight matrix
        n_members : int
            The number of ensemble members
        initialize : bool, optional
            Whether to initialize the weights, by default True
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        out_dim : int, optional
            The output dimension of the LoRA layer, by default None
        chunk_size : int, optional
            The chunk size for the vmap function, by default None
            If None all members are processed in parallel, otherwise the chunk size is used
            If 1 all members are processed sequentially, like a for loop
        """

        super().__init__()

        # If out_dim is not set, set it to dim
        if out_dim is None:
            out_dim = dim

        # Initialize all the LoRA models
        self.lora_models = [LoRA(w, rank, dim, initialize, init_type, init_settings, out_dim).to(DEVICE)
                            for _ in range(n_members)]
        
        # Set the output dimension
        self.out_dim = out_dim

        # Stack the module state
        self.params, self.buffers = stack_module_state(self.lora_models)

        # Set the number of members
        self.n_members = n_members

        # Set the base model
        self.base_model = copy.deepcopy(self.lora_models[0])
        self.base_model = self.base_model.to('meta')

        # Set base model to not require gradients
        # This can be done because meta tensors do not carry weights, they only include the model structure
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.chunk_size = chunk_size

    def _functional_call(
            self,
            x: Tensor,
            params: Dict[str, Tensor],
            buffers: Dict[str, Tensor],
    ) -> callable:
        """
        Function to call the LoRA models per member with their own parameters and buffers
        as well as their own input.

        Parameters
        ----------
        x : Tensor
            The input tensor
        params : Dict[str, Tensor]
            The parameters of the LoRA models
        buffers : Dict[str, Tensor]
            The buffers of the LoRA models

        Returns
        -------
        callable
            The functional call for the mapping of values to LoRA Models
        """

        return functional_call(self.base_model, (params, buffers), x)

    def forward(self, x):
        """
        Forward pass for the LoRA Ensemble module

        Parameters
        ----------
        x : Tensor
            The input tensor

        Returns
        -------
        out : Tensor
            The output tensor
        """
        # Extract the necessary dimensions
        sequence_length = x.shape[0]
        batch_size = x.shape[1] // self.n_members

        # Reshape the input tensor to have the ensemble members as the first dimension
        ensemble_input = x.view(sequence_length, batch_size, self.n_members, -1)
        ensemble_input = ensemble_input.movedim(2, 0)

        # Call the actual models
        out = vmap(self._functional_call, chunk_size=self.chunk_size)(ensemble_input, self.params, self.buffers)

        # Move the dimensions back to the original shape
        out = out.movedim(0, 2)
        out = out.contiguous().view(sequence_length, -1, self.out_dim)

        return out


class ASTEnsembleLoRA(EnsembleLoRA):
    """
    Class to ensemble the LoRA layers for AST.
    This just overrides the forward pass to permute the channels to match the ViT implementation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # Permute channels to match ViT Implementation
        x = x.permute(1, 0, 2)

        # Call the original forward method from EnsembleLoRA
        out = super().forward(x)

        # Permute channels back to continue pass through AST
        out = out.permute(1, 0, 2)

        return out
