#!/usr/bin/env python

"""
Implements a LoRA-Ensemble model
"""


### IMPORTS ###
# Built-in imports
import enum
from typing import List, Dict

# Lib imports
import torch
from torch import nn, Tensor

# Custom imports
from models.lora import Init_Weight, EnsembleLoRA
from models.vision_transformer import VisionTransformer, EnsembleHead, Init_Head


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### CLASS DEFINITION ###
class BatchMode(enum.Enum):
    REPEAT = 0  # Repeat the data for each ensemble member
    SPLIT = 1  # Split the data between the ensemble members
    DEFAULT = REPEAT


class LoRAEnsemble(nn.Module):
    def __init__(
            self,
            vit_model: VisionTransformer,
            rank: int,
            n_members: int,
            lora_type: str = "qkv",
            lora_layers: List[int] = None,
            lora_init: Init_Weight = Init_Weight.DEFAULT,
            init_settings: dict = None,
            batch_mode: BatchMode = BatchMode.DEFAULT,
            init_head: Init_Head = Init_Head.DEFAULT,
            head_settings: dict = None
    ) -> None:
        """
        A LoRA-Ensemble model

        Parameters
        ----------
        vit_model : VisionTransformer
            The Vision Transformer model to be used
        rank : int
            The rank of the Low Rank Adaptation module
        n_members : int
            The number of ensemble members
        lora_type : str, optional
            The projection weights to apply LoRA to.
            Can include q, k, v, by default "qkv"
        lora_layers : List[int], optional
            The layers to apply LoRA to.
            If None, apply LoRA to all layers, by default None
        lora_init : Init_Weight, optional
            Initialization method for the LoRA module, by default Init_Weight.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        batch_mode : BatchMode, optional
            The batch mode to use, by default BatchMode.DEFAULT
            This encodes whether the data is repeated in the batch dimension
            to train all ensemble members on the same data or if the data is split between the ensemble members.
        """

        # Call the super constructor
        super(LoRAEnsemble, self).__init__()

        # Set Vision Transformer
        self.vit_model = vit_model

        # Replace the VisionTransformer head with an Ensemble Head
        in_features = self.vit_model.model.heads.head.in_features
        n_classes = self.vit_model.model.heads.head.out_features
        self.vit_model.model.heads.head = EnsembleHead(in_features, n_classes, n_members, init_head, head_settings)

        # Define the batch mode
        self.batch_mode = batch_mode  # The way batch parallelization is used through the ensemble

        # Set properties
        self.n_members = n_members  # Number of ensemble members
        self.lora_type = lora_type  # Which projections LoRA is applied to

        # Set the layers to apply LoRA to
        if lora_layers is None:
            self.lora_layers = list(range(len(self.vit_model.model.encoder.layers)))
        else:
            self.lora_layers = lora_layers

        # Freeze Vision Transformer weights
        for param in self.vit_model.parameters():
            param.requires_grad = False

        # Apply LoRA to the specified layers
        for layer_id, enc_layer in enumerate(self.vit_model.model.encoder.layers):
            # If layer should not include LoRA, skip
            if layer_id not in self.lora_layers:
                continue

            # Extract dimensions for the projections of the layer
            for char in lora_type:
                if char != "q":
                    dim = getattr(enc_layer.self_attention, f"{char}dim")
                else:
                    dim = enc_layer.self_attention.embed_dim

                # Replace the original projection layers with the LoRA layers
                setattr(enc_layer.self_attention, f"{char}_proj",
                        EnsembleLoRA(
                            getattr(enc_layer.self_attention, f"{char}_proj"),
                            rank=rank,
                            dim=dim,
                            n_members=n_members,
                            initialize=True,
                            init_type=lora_init,
                            init_settings=init_settings
                        )
                        )
            dim = enc_layer.self_attention.embed_dim
            setattr(enc_layer.self_attention, "out_proj",
                    EnsembleLoRA(
                        getattr(enc_layer.self_attention, "out_proj"),
                        rank=rank,
                        dim=dim,
                        n_members=n_members,
                        initialize=True,
                        init_type=lora_init,
                        init_settings=init_settings
                    )
                    )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        ------
        out :
        """

        # If batch mode is set to repeat, repeat the data for each ensemble member
        if self.batch_mode == BatchMode.REPEAT:
            x = x.repeat_interleave(self.n_members, dim=0)

        # Run the forward pass
        out = self.vit_model(x)

        # Split the predictions into predictions for each ensemble member along the batch dimension
        # This is done independently of batch mode, as batch mode split means each ensemble member
        # is trained on a different part of the batch
        out = out.view(out.shape[0] // self.n_members, self.n_members, -1)
        out = out.permute(1, 0, 2)

        return out

    def gather_params(self):
        """
        Gather the parameters of the model.
        This dict needs to be passed to the optimizer to train the model.

        Returns
        -------
        params : Dict[str, Tensor]
            The parameters of the model
        """

        params = {}

        # Gather the LoRA parameters
        for layer_id, enc_layer in enumerate(self.vit_model.model.encoder.layers):
            # If layer should not include LoRA, skip
            if layer_id not in self.lora_layers:
                continue

            for char in self.lora_type:
                # Gather the parameters of the LoRA module
                proj_params = enc_layer.self_attention.__getattr__(f"{char}_proj").params
                proj_params = {f"encoder_layer_{layer_id}_{char}_proj_{k}": v for k, v in proj_params.items()
                               if k in ["w_a.weight", "w_b.weight"]}
                params.update(proj_params)

            # Gather the parameters of the out projection
            out_proj_params = enc_layer.self_attention.out_proj.params
            out_proj_params = {f"encoder_layer_{layer_id}_out_proj_{k}": v for k, v in out_proj_params.items()
                               if k in ["w_a.weight", "w_b.weight"]}
            params.update(out_proj_params)

        # Add parameters of the add
        for name, param in self.vit_model.model.heads.head.params.items():
            params.update({f"head_{name}": param})

        # Add other trainable parameters from the model
        for name, param in self.vit_model.named_parameters():
            if param.requires_grad:
                params.update({name: param})

        return params

    def set_params(self, model_state_dict: Dict[str, Tensor]):
        """
        Set the parameters of the model based on a model state dict

        Parameters
        ----------
        model_state_dict : Dict[str, Tensor]
            The model state dict to set the parameters from
        """

        # Set the parameters of the model
        for key, value in model_state_dict.items():
            # Set for encoder layers
            if 'encoder' in key:
                # Get the layer and projection
                layer_name = "_".join(key.split("_")[:3])
                layer = self.vit_model.model.encoder.layers.__getattr__(layer_name)
                proj_name = "_".join(key.split("_")[3:5])
                #proj = layer.self_attention.__getattr__(proj_name)
                self.vit_model.model.encoder.layers.__getattr__(layer_name).self_attention.__getattr__(proj_name).params["_".join(key.split("_")[5:])] = value

                # Set the parameter
                #proj.__setattr__("_".join(key.split("_")[5:]), value)
            # Set for head (legacy without ensemble head)
            elif 'heads.head' in key:
                # Catch legacy model with trainable base_model parameters
                if 'base_model' in key:
                    continue
                # Set the parameter for the head
                self.vit_model.model.heads.head.__setattr__(key.split(".")[-1], value)
            # Set for ensemble head
            elif 'head' in key:
                self.vit_model.model.heads.head.params[key.split("_")[-1]] = value
                #self.vit_model.model.heads.head.params.__setattr__(key.split("_")[-1], value)
            # Set for other trainable parameters
            else:
                self.vit_model.__setattr__(key, value)
