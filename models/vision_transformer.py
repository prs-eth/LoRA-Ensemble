#!/usr/bin/env python

"""
Implements the vision transformer models used for this project
"""

### IMPORTS ###
# Built-in imports
from typing import Tuple, Optional, Dict, List
import copy
import enum

import numpy as np
# Lib imports
import torchvision.models
import torch
from torch import nn, Tensor, vmap
from torch.func import stack_module_state, functional_call
from torchvision.models import vision_transformer as torch_vit
import torch.nn.functional as F

# Custom imports
from models.lora import EnsembleLoRA
from utils_GPU import DEVICE


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### STATIC FUNCTIONS ###
def validate_model_config(config: str, patch_size: int) -> None:
    """
    Validate the model configration passed to the constructor

    Parameters
    ----------
    config : String
        The model configuration
    patch_size : Integer
        The patch size for the model
    """

    valid_patch = False  # Set the flag for valid patch size
    # Check if patch_size is available with the given config and if config exists
    if config == "base" or config == "large":
        if patch_size in [16, 32]:
            valid_patch = True
    elif config == "huge":
        if patch_size == 14:
            valid_patch = True
    else:
        # If config is not available, raise a ValueError
        raise ValueError(f"Model config needs to be one of [\"base\", \"large\", \"huge\"]. Got {config}.")

    if not valid_patch:
        # If the patch size is not available with the config, raise a ValueError
        raise ValueError("Illegal patch size.")


### CLASS DEFINITION ###
class Init_Head(enum.Enum):
    NORMAL = 0
    KAIMING_UNIFORM = 1
    XAVIER_UNIFORM = 2
    DEFAULT = 3


class VisionTransformer(nn.Module):
    def __init__(
            self,
            n_classes: int,
            config: str,
            patch_size: int,
            pretrained: bool = True,
            weight_type: str = None,
            custom_attention: bool = True,
            init_head: Init_Head = Init_Head.DEFAULT,
            init_settings: dict = None
    ) -> None:
        """
        A Vision Transformer model

        Parameters
        ----------
        n_classes : int
            The number of classes for the classification task
        config : str
            The configuration of Vision Transformer to use
            One of {"base", "large", "huge"}
        patch_size : int
            The size of the patches
            One of
            {
                "base" : (16, 32),
                "large" : (16, 32),
                "huge": 14
            }
            NOTE: Smaller patch size means higher computational cost
        pretrained : bool, optional
            Whether to use a pretrained model, by default True
        weight_type : str, optional
            The weight type that should be used.
            If pretrained is True and weight_type is None, the default weights are used.
            Please refer to the torchvision documentation for available weights.
            https://pytorch.org/vision/main/models/vision_transformer.html
        custom_attention : bool, optional
            Whether to use a custom multihead attention module, by default True
        init_head : Init_Head, optional
            The type of initialization to use for the head, by default Init_Head.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        super(VisionTransformer, self).__init__()

        # Validate the model input
        validate_model_config(config, patch_size)

        # Set internal parameters
        self._pretrained = pretrained
        self._config = config
        self._patch_size = patch_size
        self._weight_type = weight_type

        # Load the model architecture and weights
        vit_model, vit_weights = self.load_model()

        # Define model
        self.weights = vit_weights
        self.model = vit_model(weights=vit_weights)

        # Change number of classes
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, n_classes)

        # Initialize the head
        self.initialize_head(init_head, init_settings)

        # Set the flag whether to use custom attention
        self._custom_attention = custom_attention

        # If custom attention is enabled set it up
        if self._custom_attention:
            self.get_custom_attention()

    @property
    def config(self) -> str:  # Return the model config
        return self._config

    @property
    def patch_size(self) -> int:  # Return the patch size
        return self._patch_size

    @property
    def custom_attention(self) -> bool:  # Return the custom attention flag
        return self._custom_attention

    def load_model(self) -> Tuple[nn.Module, nn.Module]:
        """
        Loads the defined model based on the information in the constructor

        Returns
        -------
        vit_model : Torchvision model
            The loaded model
        vit_weights : Torchvision weights or None
            The loaded pretrained weights.
            None if pretrained is False.
        """

        # Construct the model string
        model_string = f"vit_{self._config[0]}_{self._patch_size}"

        # Import the model
        vit_model = getattr(torch_vit, model_string)

        # Load weights if pretrained
        if self._pretrained:
            try:
                # Construct the weight string
                if self._weight_type is None:
                    weight_type = "DEFAULT"
                else:
                    weight_type = self._weight_type

                weight_string = f"ViT_{self._config[0].capitalize()}_{self._patch_size}_Weights"  # .{weight_type}"

                # Load the weights
                vit_weights_enum = getattr(torch_vit, weight_string)
                vit_weights = vit_weights_enum[weight_type]
            except AttributeError:
                # Catch missing weights, raise a ValueError
                raise ValueError(f"No weights {self._weight_type} are available for model {model_string}.\n"
                                 f"Available weights are {torchvision.models.get_model_weights(model_string)}")

        else:
            # If not pretrained, set weights to None
            vit_weights = None

        return vit_model, vit_weights

    def preprocess(self) -> callable:
        """
        Preprocess the input data for the Vision Transformer model

        Returns
        -------
        transforms : callable
            The preprocessing function as defined by the model weights
        """

        # Get the transformation for the imported model
        transforms = self.weights.transforms()

        return transforms

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Vision Transformer model

        Parameters
        ----------
        x : Tensor
            The input data

        Returns
        -------
        out = Tensor
            The output of the model
        """

        out = self.model(x)

        return out

    def get_weights(self) -> dict:
        """
        Returns the named parameters of the model

        Returns
        -------
        weights : Dictionary
            The named parameters of the model
        """

        # Extract the weights of the model
        weights = self.model.named_parameters()

        return weights

    def get_custom_attention(self):
        """
        Replace original multihead attention module with custom multihead attention module.
        Weigthts and biases of original multihead attention module are loaded into the custom multihead
        attention module.
        """

        # Replace self attention with new attention module
        for name, module in self.model.named_modules():

            # Modify only self attention modules that have not already been replaced
            if name.endswith("self_attention") and not isinstance(module, MultiHeadAttention):

                # Get weights and biases of original self attention module
                if module.in_proj_weight is not None:
                    # If weights are merged get the complete in_proj_weights
                    in_proj_weight = module.in_proj_weight
                    # Set the rest to None
                    q_proj_weight = None
                    k_proj_weight = None
                    v_proj_weight = None
                else:
                    # If weights are not merged set complete in_proj_weights to None
                    in_proj_weight = None
                    # Get the rest of the weights
                    q_proj_weight = module.q_proj_weight
                    k_proj_weight = module.q_proj_weight
                    v_proj_weight = module.v_proj_weight

                # Get the out_proj_weights
                out_proj_weight = module.out_proj.weight

                # Get the biases
                if module.in_proj_bias is not None:
                    # If biases are used set flag and extract them
                    bias = True
                    in_proj_bias = module.in_proj_bias
                    out_proj_bias = module.out_proj.bias
                else:
                    # Set flag and set biases to None
                    bias = False
                    in_proj_bias = None
                    out_proj_bias = None

                # Check if bias_kv is used
                if module.bias_k is not None and module.bias_v is not None:
                    # If k and v have bias set flag and extract the values
                    add_bias_kv = True
                    bias_k, bias_v = module.bias_k, module.bias_v
                else:
                    # Set flag and set biases to None
                    add_bias_kv = False
                    bias_k, bias_v = None, None

                # Replace self attention module with new attention module
                multihead = MultiHeadAttention(embed_dim=module.embed_dim, num_head=module.num_heads,
                                               dropout=module.dropout, bias=bias, add_bias_kv=add_bias_kv,
                                               kdim=module.kdim, vdim=module.vdim,
                                               batch_first=module.batch_first)

                # Set the weights of the new multihead attention module
                multihead.set_loaded_weights(out_proj_weight=out_proj_weight, qkv_same_dim=True,
                                             in_proj_weight=in_proj_weight, q_proj_weight=q_proj_weight,
                                             k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight,
                                             in_proj_bias=in_proj_bias, out_proj_bias=out_proj_bias,
                                             bias_k=bias_k, bias_v=bias_v)

                # Replace old attention module with new attention module
                parent_module_name = name.rsplit(".", 1)[0]
                parent_module = self.model
                for part in parent_module_name.split("."):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, "self_attention", multihead)

    def initialize_head(self, init_type: Init_Head = Init_Head.DEFAULT, init_settings: dict = None) -> None:
        """
        Initialize the weights of the LoRA matrices

        Parameters
        ----------
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        if init_type == Init_Head.NORMAL:
            # If no settings are given set default
            if init_settings is None:
                init_settings = {
                    "mean": 0,
                    "std": 0.02
                }

            # Initialize the weights
            nn.init.normal_(self.model.heads.head.weight, **init_settings)

        elif init_type == Init_Head.KAIMING_UNIFORM:
            # If no settings are given set default
            if init_settings is None:
                init_settings = {
                    "a": np.sqrt(5)
                }

            # Initialize the weights
            nn.init.kaiming_uniform_(self.model.heads.head.weight, **init_settings)

        elif init_type == Init_Head.XAVIER_UNIFORM:
            # If no settings are given set default
            if init_settings is None:
                init_settings = {
                    "gain": 1
                }

            # Initialize the weights
            nn.init.xavier_uniform_(self.model.heads.head.weight, **init_settings)

        elif init_type == Init_Head.DEFAULT:
            # Use the default initialization of the model
            pass

        else:
            raise ValueError("Invalid initialization type")


class MultiHeadAttention(nn.Module):
    """
    Custom MultiHeadAttention module, to allow for LoRA to be added to the model,
    as this is not possible with the default PyTorch implementation.

    The constructor and forward pass function signature are taken from the PyTorch implementation.
    Most of the properties and forward pass implementation are copied from the PyTorch implementation.
    This is done to allow insertion of this class into pretrained PyTorch models, by replacing their MultiHeadAttention
    implementation with this one.

    The set_loaded_weights function is added to allow setting model weights from pretrained models.
    The implementation is such that it is compatible with the PyTorch implementation.
    """

    # NOTE: If there are errors with needing the other functions the PyTorch implementation offers make this
    # a subclass of nn.modules.activation.MultiHeadAttention and overload constructor and forward pass

    def __init__(self, embed_dim, num_head, dropout=0.0, bias=True, add_bias_kv=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        """
        Initialize the Multi-Headed Attention module

        Parameters
        ----------
        embed_dim : int
            Total dimension of the model
        num_head : int
            Number of attention heads. Embed_dim will be split across attention heads.
        dropout : float, optional
            Dropout probability on attention weights, by default 0.0
        bias : bool, optional
            Whether to add bias to the input and output projection layers, by default True
        add_bias_kv : bool, optional
            Whether to add bias to the key and value sequences, by default False
        kdim : int, optional
            Number of features in key, by default None. Uses embed_dim if None
        vdim : int, optional
            Number of features in value, by default None. Uses embed_dim if None
        batch_first : bool, optional
            Whether the input and output tensors are provided in batch_first format, by default False

        Credit
        ------
        Documentation and function signature largely taken from the PyTorch implementation.
        Most of the code and properties are copied from the PyTorch implementation.
        """

        # Call super constructor
        super(MultiHeadAttention, self).__init__()

        # Set the embedding dimensions according to the torch implementation
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Define number of heads
        self.num_head = num_head

        # Set dropout
        self.dropout = dropout

        self.batch_first = batch_first

        # Set the dimensions of each head
        self.head_dim = embed_dim // num_head
        assert self.head_dim * num_head == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Set the bias flags
        self.bias = bias
        self.add_bias_kv = add_bias_kv

        # Define the linear layers
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.kdim, bias=self.add_bias_kv, device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.vdim, bias=self.add_bias_kv, device=device, dtype=dtype)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_casual: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of the Multi-Headed Attention module

        query : Tensor
            Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key : Tensor
            Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value : Tensor
            Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask : Tensor, optional
            If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights : bool, optional
            If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
            and achieve the best performance for MHA.
            Default: ``True``.
        attn_mask : Tensor, optional
            If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcast across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        average_attn_weights : bool, optional
            If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal : bool, optional
            If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

        Returns
        -------
        attn_output: Tensor
            Attention outputs of shape :math:`(L, E)` when input is unbatched,
            :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
            where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
            embedding dimension ``embed_dim``.
        attn_output_weights: Tensor, optional
            Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
            returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
            :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
            :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
            head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        Credit
        ------
        Documentation and function signature taken from the PyTorch documentation.
        """

        # Check if data is batched
        is_batched = query.dim() == 3

        # Set the masks if necessary
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # Check for nested tensors. This is also not supported by PyTorches basic implementation
        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, "This MultiheadAttention implementation does not support NestedTensor"

        if self.batch_first and is_batched:
            # Make sure transpose does not affect is operation
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # If data is unbatched unsqueeze at batch dimension to ensure they have same dimension as batched inputs
        # Before returning the data is squeezed again to ensure output dimension matches expectation
        if not is_batched:
            query.unsqueeze(1)
            key.unsqueeze(1)
            value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask.unsqueeze(1)

        # Run the input projections
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)

        # Get the necessary shape variables
        target_len, batch_size, _ = query.shape
        source_len = key.shape[0]

        # Ensure correct shape of the attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (target_len, source_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The size of the 2D attn_mask is expected to be {correct_2d_size}, but got {attn_mask.shape}")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (batch_size * self.num_head, target_len, source_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The size of the 3D attn_mask is expected to be {correct_3d_size}, but got {attn_mask.shape}")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # If bias is present pad the mask to include the bias column in the weight matrix
        if self.add_bias_kv:
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # Create views for all heads
        q = q.view(target_len, batch_size * self.num_head, self.head_dim).transpose(0, 1)
        k = k.view(source_len, batch_size * self.num_head, self.head_dim).transpose(0, 1)
        v = v.view(source_len, batch_size * self.num_head, self.head_dim).transpose(0, 1)

        source_len = k.size(1)

        # Ensure proper masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, source_len), \
                f"expecting key_padding_mask shape of {(batch_size, source_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, source_len). \
                expand(-1, self.num_head, -1, -1).reshape(batch_size * self.num_head, 1, source_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # Disable dropout if inference
        if not self.training:
            dropout = 0.0
        else:
            dropout = self.dropout

        # Ensure proper dimension of attention mask
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(batch_size, self.num_head, -1, source_len)

        # Create views pe head
        q = q.view(batch_size, self.num_head, target_len, self.head_dim)
        k = k.view(batch_size, self.num_head, source_len, self.head_dim)
        v = v.view(batch_size, self.num_head, source_len, self.head_dim)

        # Get attention scores using the original scaled dot product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout, is_casual)
        # Reset shape of attention output to proper shape
        is_ensemble_lora = isinstance(self.out_proj, EnsembleLoRA)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()

        # If model is a LoRA Ensemble need to be able to recover the batch dimension separately
        if is_ensemble_lora:
            attn_output = attn_output.view(target_len, batch_size, self.embed_dim)
        else:
            attn_output = attn_output.view(batch_size * target_len, self.embed_dim)

        # Perform output projection
        attn_output = self.out_proj(attn_output)

        # If model is a LoRA Ensemble batch dimension and target length were never collapsed
        if not is_ensemble_lora:
            attn_output = attn_output.view(target_len, batch_size, attn_output.size(1))

        # If data was not batched, squeeze again to match expected output shape
        if not is_batched:
            attn_output.squeeze(1)

        # If batch dimension was first dimension set it as such again
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, None

    def set_loaded_weights(
            self,
            out_proj_weight: Tensor,
            qkv_same_dim: bool,
            in_proj_weight: Optional[Tensor] = None,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            in_proj_bias: Optional[Tensor] = None,
            out_proj_bias: Optional[Tensor] = None,
            bias_k: Optional[Tensor] = None,
            bias_v: Optional[Tensor] = None,
    ) -> None:
        """
        Set the loaded weights to the model

        Parameters
        ----------
        out_proj_weight : Tensor
            The output projection weights
        qkv_same_dim : bool
            Whether the query, key, and value projections have the same dimension
        in_proj_weight : Tensor, optional
            The input projection weights, by default None
            Must be provided if qkv_same_dim is True
        q_proj_weight : Tensor, optional
            The query projection weights, by default None
            Must be provided if qkv_same_dim is False
        k_proj_weight : Tensor, optional
            The key projection weights, by default None
            Must be provided if qkv_same_dim is False
        v_proj_weight : Tensor, optional
            The value projection weights, by default None
            Must be provided if qkv_same_dim is False
        in_proj_bias : Tensor, optional
            The input projection bias, by default None
            Must be provided if self.bias is True
        out_proj_bias : Tensor, optional
            The output projection bias, by default None
            Must be provided if self.bias is True
        bias_k : Tensor, optional
            The key projection bias, by default None
            Must be provided if self.add_bias_kv is True
        bias_v : Tensor, optional
            The value projection bias, by default None
            Must be provided if self.add_bias_kv is True
        """

        # If query, key and value have the same dimension, the in_proj_weight is merged
        if qkv_same_dim:
            # Check correct input
            if in_proj_weight is None:
                raise ValueError("in_proj_weight must be provided if qkv_same_dim is True")

            # Split in_proj_weight into three separate projection weights
            q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weight.chunk(3)

            # If there bias is used, split the bias into three parts
            if self.bias:
                q_proj_bias, k_proj_bias, v_proj_bias = in_proj_bias.chunk(3)
        else:
            # If query, key and value have different dimensions, the weights are provided separately
            if any([q_proj_weight is None, k_proj_weight is None, v_proj_weight is None]):
                # Ensure all weights are provided
                raise ValueError(
                    "q_proj_weight, k_proj_weight, and v_proj_weight must be provided if qkv_same_dim is False")

        # Check if bias is used and if so if it is provided
        if self.bias and any([in_proj_bias is None, out_proj_bias is None]):
            raise ValueError("in_proj_bias and out_proj_bias must be provided if self.bias is True")

        # Check if k and v have bias and if so if it is provided
        if self.add_bias_kv and any([bias_k is None, bias_v is None]):
            raise ValueError("k_bias and v_bias must be provided if self.add_bias_kv is True")

        # Set the new weights without adding gradient connections
        with torch.no_grad():
            # Set the weights
            self.q_proj.weight = nn.Parameter(q_proj_weight)
            self.k_proj.weight = nn.Parameter(k_proj_weight)
            self.v_proj.weight = nn.Parameter(v_proj_weight)
            self.out_proj.weight = nn.Parameter(out_proj_weight)

            # Set the biases if used
            if self.bias:
                self.q_proj.bias = nn.Parameter(q_proj_bias)
                self.k_proj.bias = nn.Parameter(k_proj_bias)
                self.v_proj.bias = nn.Parameter(v_proj_bias)
                self.out_proj.bias = out_proj_bias

            # Set the k and v biases if used
            if self.add_bias_kv:
                self.k_proj.bias = bias_k
                self.v_proj.bias = bias_v


class VisionTransformerEnsemble(nn.Module):
    """
    A class for a vision transformer ensemble
    """

    def __init__(
            self,
            n_members: int,
            n_classes: int,
            config: str,
            patch_size: int,
            pretrained: bool = True,
            weight_type: str = None,
            perturb_scale: float = None,
            max_perturb_layer: int = None,
            init_head: Init_Head = Init_Head.DEFAULT,
            init_settings: dict = None
    ):
        """
        Initialize the Vision Transformer Ensemble
        Parameters
        ----------
        n_members : int
            The number of ensemble members
        n_classes : int
            The number of classes for the classification task
        config : str
            The configuration of Vision Transformer to use
            One of {"base", "large", "huge"}
        patch_size : int
            The size of the patches
            One of
            {
                "base" : (16, 32),
                "large" : (16, 32),
                "huge": 14
            }
            NOTE: Smaller patch size means higher computational cost
        pretrained : bool, optional
            Whether to use a pretrained model, by default True
        weight_type : str, optional
            The weight type that should be used.
            If pretrained is True and weight_type is None, the default weights are used.
            Please refer to the torchvision documentation for available weights.
            https://pytorch.org/vision/main/models/vision_transformer.html
        perturb_scale : float, optional
            The scale of the weight perturbation. This is a factor that is multiplied with the standard deviation of
            the respective weight matrices. If None, no perturbation is applied. By default None
        max_perturb_layer : int, optional
            The maximum number of layers that should be perturbed. If None, all layers are perturbed. By default None
        init_head : Init_Head, optional
            The type of initialization to use for the head, by default Init_Head.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        super(VisionTransformerEnsemble, self).__init__()

        self.n_members = n_members

        # Create the list of ensemble members
        self.vit_models = [VisionTransformer(n_classes, config, patch_size, pretrained, weight_type,
                                             custom_attention=False, init_head=init_head,
                                             init_settings=init_settings).to(DEVICE)
                           for _ in range(self.n_members)]

        # Perturb the weights
        apply_to_list = ["in_proj", "out_proj", "mlp"]  # List of weight matrices to apply perturbation to
        # If not perturbation scale is None or 0 do not perturb the weights
        if perturb_scale is not None and perturb_scale != 0:
            # Get number of encoder layers
            num_layers = len(list(self.vit_models[0].model.encoder.layers))
            # If max_perturb_layer is None, perturb all layers
            if max_perturb_layer is None:
                max_perturb_layer = num_layers
            # If max_perturb_layer is 0, do not perturb any layers
            if max_perturb_layer != 0:
                # Perturb the weights
                for vit_model in self.vit_models:
                    # Execute for each encoder layer
                    for i, layer in enumerate(reversed(list(vit_model.model.encoder.layers))):
                        # Limit to the first max_perturb_layer layers
                        if i >= num_layers - max_perturb_layer:
                            break
                        # Apply to the specified weight matrices only if they are trainable
                        for name, param in layer.named_parameters():
                            if any([param_name in name for param_name in apply_to_list]):
                                if param.requires_grad:
                                    param_std = torch.std(param)
                                    param.data += torch.randn_like(param) * perturb_scale * param_std

    # DEPRECATED: Due to an issue with the Torchvision ViT model this is not used
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

        raise DeprecationWarning("Due to an issue with the Torchvision ViT model this is not used")

        return functional_call(self.base_model, (params, buffers), (x,))

    def forward(self, x: Tensor) -> Tensor:

        out = [model(x) for model in self.vit_models]
        out = torch.stack(out)

        return out

    def set_params(self, model_state_list: List[Tensor]) -> None:
        """
        Set the parameters of the ensemble members

        Parameters
        ----------
        model_state_list : List[Tensor]
            The parameters to set
        """

        param_index = 0
        for member in self.vit_models:
            length_model_in_list = int(len(model_state_list) / len(self.vit_models))
            model_params = model_state_list[param_index:param_index + length_model_in_list]
            count = 0
            for p in member.parameters():
                p.data = model_params[count]
                count += 1
            param_index += length_model_in_list

        #raise NotImplementedError("This function is not implemented yet")

        #model_length = len(model_state_list) // self.n_members
        #for i, model in enumerate(self.vit_models):
        #    for j, param in enumerate(model.parameters()):
        #        param = model_state_list[i * model_length + j]
        #        pass


class EnsembleHead(nn.Module):
    """
    A class to handle multiple output layers for an ensemble
    """

    def __init__(
            self,
            in_features: int,
            n_classes: int,
            n_members: int,
            init_type: Init_Head = Init_Head.DEFAULT,
            init_settings: dict = None,
            layer_norm: bool = False
    ):
        """
        Constructor for the EnsembleHead class

        Parameters
        ----------
        in_features : int
            The number of input features
        n_classes : int
            The number of classes
        n_members : int
            The number of ensemble members
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        super(EnsembleHead, self).__init__()

        # Create the list of ensemble members
        self._layer_norm = layer_norm
        if not self._layer_norm:
            # If layer norm is not used, create the heads as a list of linear layers
            self.heads = [nn.Linear(in_features, n_classes).to(DEVICE) for _ in range(n_members)]
        else:
            # If layer norm is used, create the heads as a list of layer norm and linear layers
            self.heads = [
                nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, n_classes)
                ).to(DEVICE) for _ in range(n_members)
            ]

        # Initialize the model weights
        self.initialize_weights(init_type, init_settings)

        # Stack the model states
        self.params, self.buffers = stack_module_state(self.heads)

        # Give parameters clear names if layer norm is included
        if self._layer_norm:
            for name in list(self.params.keys()):
                if "0" in name:
                    new_name = name.replace("0", "ln")
                else:
                    new_name = name.replace("1", "linear")
                self.params[new_name] = self.params.pop(name)

        # Set the number of members
        self.n_members = n_members

        # Set the base model
        self.base_model = copy.deepcopy(self.heads[0])
        self.base_model = self.base_model.to('meta')

        # Set base model to not require gradients
        # This can be done because meta tensors do not carry weights, they only include the model structure
        for param in self.base_model.parameters():
            param.requires_grad = False

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
        batch_size = x.shape[0] // self.n_members

        # Reshape the input tensor to have the ensemble members as the first dimension
        ensemble_input = x.view(batch_size, self.n_members, -1)
        ensemble_input = ensemble_input.movedim(1, 0)

        # Call the actual models
        out = vmap(self._functional_call)(ensemble_input, self.params, self.buffers)

        # Move the dimensions back to the original shape
        out = out.movedim(0, 1)
        out = out.contiguous().view(batch_size * self.n_members, -1)

        return out

    def initialize_weights(self, init_type: Init_Head = Init_Head.DEFAULT, init_settings: dict = None) -> None:
        """
        Initialize the weights of the ensemble members

        Parameters
        ----------
        init_type : INIT_WEIGHT, optional
            The type of initialization to use, by default INIT_WEIGHT.DEFAULT
        init_settings : dict, optional
            Settings for the initialization method, by default None
        """

        for head in self.heads:
            if self._layer_norm:
                head = head[1]
            if init_type == Init_Head.NORMAL:
                # If no settings are given set default
                if init_settings is None:
                    init_settings = {
                        "mean": 0,
                        "std": 0.02
                    }

                # Initialize the weights
                nn.init.normal_(head.weight, **init_settings)

            elif init_type == Init_Head.KAIMING_UNIFORM:
                # If no settings are given set default
                if init_settings is None:
                    init_settings = {
                        "a": np.sqrt(5)
                    }

                # Initialize the weights
                nn.init.kaiming_uniform_(head.weight, **init_settings)

            elif init_type == Init_Head.XAVIER_UNIFORM:
                # If no settings are given set default
                if init_settings is None:
                    init_settings = {
                        "gain": 1
                    }

                # Initialize the weights
                nn.init.xavier_uniform_(head.weight, **init_settings)

            elif init_type == Init_Head.DEFAULT:
                # Use the default initialization of the model
                pass

            else:
                raise ValueError("Invalid initialization type")
