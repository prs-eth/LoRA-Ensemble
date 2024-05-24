#!/usr/bin/env python

"""
Implements a Vision Transformer model with a Low Rank Adaptation module
"""


### IMPORTS ###
# Built-in imports
from typing import List

# Lib imports
from torch import nn, Tensor

# Custom imports
from models.lora import LoRA, Init_Weight


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### CLASS DEFINITION ###
class LoRAViT(nn.Module):
    def __init__(
            self,
            vit_model: nn.Module,
            rank: int,
            lora_type: str = "qkv",
            lora_layers: List[int] = None,
            lora_init: Init_Weight = Init_Weight.DEFAULT
    ) -> None:
        """
        A Vision Transformer model with a Low Rank Adaptation module

        Parameters
        ----------
        vit_model : nn.Module
            The Vision Transformer model
        rank : int
            The rank of the Low Rank Adaptation module
        lora_type : str, optional
            The projection weights to apply LoRA to.
            Can include q, k, v, by default "qkv"
        lora_layers : List[int], optional
            The layers to apply LoRA to.
            If None, apply LoRA to all layers, by default None
        lora_init : Init_Weight, optional
            Initialization method for the LoRA module, by default Init_Weight.DEFAULT
        """

        # Call the super constructor
        super(LoRAViT, self).__init__()

        # Define Vision Transformer
        self.vit_model = vit_model

        # Set layers to apply LoRA to
        if lora_layers is None:
            self.lora_layers = list(range(len(self.vit_model.model.encoder.layers)))
        else:
            self.lora_layers = lora_layers

        # Freeze Vision Transformer weights
        for param in self.vit_model.parameters():
            param.requires_grad = False

        # make head trainable
        for param in self.vit_model.model.heads.head.parameters():
            param.requires_grad = True

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
                        LoRA(
                            getattr(enc_layer.self_attention, f"{char}_proj"),
                            rank=rank,
                            dim=dim,
                            initialize=True,
                            init_type=lora_init
                        )
                        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Vision Transformer and the Low Rank Adaptation module

        Parameters
        ----------
        x : Tensor
            The input tensor

        Returns
        -------
        out : Tensor
            The output tensor
        """

        out = self.vit_model(x)

        return out
