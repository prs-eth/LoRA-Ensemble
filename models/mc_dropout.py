#!/usr/bin/env python

from typing import Dict

import torch
from torch import nn, Tensor

from models.vision_transformer import VisionTransformer, EnsembleHead, Init_Head
from utils_GPU import DEVICE
from models.AST import ASTModel
from models.lora_ensemble import BatchMode
import torch.nn.functional as F
from timm.models.vision_transformer import Attention


class MCDropoutEnsemble(nn.Module):
    def __init__(
            self,
            vit_model: nn.Module,
            n_members: int,
            p_drop: float,
            batch_mode: BatchMode = BatchMode.DEFAULT,
            init_head: Init_Head = Init_Head.DEFAULT,
            head_settings: dict = None
    ):
        super().__init__()

        self.vit_model = vit_model

        # Use an Ensemble Head for fair comparison with LoRA ensemble
        in_features = self.vit_model.model.heads.head.in_features
        n_classes = self.vit_model.model.heads.head.out_features
        self.vit_model.model.heads.head = EnsembleHead(
            in_features, n_classes, n_members, init_head, head_settings)

        self.n_members = n_members
        self.p_drop = p_drop
        self.batch_mode = batch_mode

        self._enable_dropkey()

        # self._set_p_drop()

    def _set_p_drop(self):
        """
        Sets dropout probabilities in MLP layers.
        """
        dropouts = [m for m in self.vit_model.modules() if isinstance(m, torch.nn.Dropout)]
        for dropout in dropouts:
            dropout.p = self.p_drop

    def _enable_dropkey(self):
        """
        Enable attention dropout à la DropKey (https://arxiv.org/abs/2208.02646), independent for
        each batch element.
        """
        self_attention_layers = [
            m for n, m in self.vit_model.named_modules() if n.endswith('self_attention')]
        for layer in self_attention_layers:
            def wrapper(q, k, v, inner_forward=layer.forward, **kwargs):
                ones = torch.ones(q.shape[0] * layer.num_heads, q.shape[1], k.shape[1]).to(DEVICE)
                key_mask = torch.bernoulli(ones * self.p_drop).bool()
                return inner_forward(q, k, v, attn_mask=key_mask, **kwargs)

            layer.forward = wrapper

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_mode == BatchMode.REPEAT:
            x = x.repeat_interleave(self.n_members, dim=0)

        out = self.vit_model(x)

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

        return dict(self.vit_model.named_parameters())

    def set_params(self, model_state_dict: Dict[str, Tensor]):
        """
        Set the parameters of the model based on a model state dict

        Parameters
        ----------
        model_state_dict : Dict[str, Tensor]
            The model state dict to set the parameters from
        """

        for name, param in model_state_dict.items():
            if name in self.vit_model.state_dict():
                # If the parameter exists in the second model, load it
                self.vit_model.state_dict()[name].copy_(param)

        # load state dict
        

        #self.vit_model.load_state_dict(model_state_dict)


        # Set the parameters of the model
        #for key, value in model_state_dict.items():
        #    # Set for encoder layers
        #    if 'encoder' in key:
        #        # Get the layer and projection
        #        layer_name = "_".join(key.split("_")[:3])
        #        layer = self.vit_model.model.encoder.layers.__getattr__(layer_name)
        #        proj_name = "_".join(key.split("_")[3:5])
        #        proj = layer.self_attention.__getattr__(proj_name)

        #        # Set the parameter
        #        proj.__setattr__("_".join(key.split("_")[5:]), value)
        #    # Set for head (legacy without ensemble head)
        #    elif 'heads.head' in key:
        #        # Catch legacy model with trainable base_model parameters
        #        if 'base_model' in key:
        #            continue
        #        # Set the parameter for the head
        #        self.vit_model.model.heads.head.__setattr__(key.split(".")[-1], value)
        #    # Set for ensemble head
        #    elif 'head' in key:
        #        self.vit_model.model.heads.head.params.__setattr__(key.split("_")[-1], value)
        #    # Set for other trainable parameters
        #    else:
        #        self.vit_model.__setattr__(key, value)

class DropkeyAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mask_ratio=0.5):
        super(DropkeyAttention, self).__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.mask_ratio = mask_ratio
        
    def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            # use DropKey as regularizer
            m_r = torch.ones_like(attn) * self.mask_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12

            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    
# original Attention class of DeiT
#class Attention(nn.Module):
#    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#        super().__init__()
#        self.num_heads = num_heads
#        head_dim = dim // num_heads
#        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#        self.scale = qk_scale or head_dim ** -0.5
#
#        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#        self.attn_drop = nn.Dropout(attn_drop)
#        self.proj = nn.Linear(dim, dim)
#        self.proj_drop = nn.Dropout(proj_drop)
#
#    def forward(self, x):
#        B, N, C = x.shape
#        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
#
#        attn = (q @ k.transpose(-2, -1)) * self.scale
#        attn = attn.softmax(dim=-1)
#        attn = self.attn_drop(attn)
#
#        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#        x = self.proj(x)
#        x = self.proj_drop(x)
#        return x

class ASTMCDropoutEnsemble(nn.Module):
    def __init__(
            self,
            AST_model: nn.Module,
            n_members: int,
            p_drop: float,
            batch_mode: BatchMode = BatchMode.DEFAULT,
            init_head: Init_Head = Init_Head.DEFAULT,
            head_settings: dict = None
    ):
        super().__init__()

        self.ast_model = AST_model

 
        # Use an Ensemble Head for fair comparison with LoRA ensemble
        n_classes = self.ast_model.mlp_head[1].out_features
        self.ast_model.mlp_head = nn.Sequential(nn.LayerNorm(self.ast_model.original_embedding_dim).to(DEVICE),
                                              EnsembleHead(self.ast_model.original_embedding_dim, n_classes, n_members, init_head, head_settings))

        self.n_members = n_members
        self.p_drop = p_drop
        self.batch_mode = batch_mode

        self._enable_dropkey()

        # self._set_p_drop()

    def _set_p_drop(self):
        """
        Sets dropout probabilities in MLP layers.
        """

        raise NotImplementedError("This method is not implemented yet.")
        dropouts = [m for m in self.ast_model.modules() if isinstance(m, torch.nn.Dropout)]
        for dropout in dropouts:
            dropout.p = self.p_drop

    def _enable_dropkey(self):
        """
        Enable attention dropout à la DropKey (https://arxiv.org/abs/2208.02646), independent for
        each batch element.
        """
        for layer_id, layer in enumerate(self.ast_model.v.blocks):
            # Replace the original attention layers with the DropkeyAttention layers

            # access original weights
            original_state_dict = layer.attn.state_dict()

            # replace attention layer with DropkeyAttention
            setattr(layer, "attn",
                    DropkeyAttention(
                        layer.attn.qkv.in_features,
                        num_heads=layer.attn.num_heads,
                        qkv_bias=layer.attn.qkv.bias is not None,
                        qk_scale=layer.attn.scale,
                        attn_drop=layer.attn.attn_drop.p,
                        proj_drop=layer.attn.proj_drop.p,
                        mask_ratio=self.p_drop
                    ).to(DEVICE)
                    )
            
            # load original weights
            layer.attn.load_state_dict(original_state_dict)
            
            # If layer should not include LoRA, skip
            #if layer_id not in self.lora_layers:
            #    continue
#
            ## Extract dimensions for the projections of the layer
            #for char in lora_type:
            #    if char != "q":
            #        dim = getattr(enc_layer.self_attention, f"{char}dim")
            #    else:
            #        dim = enc_layer.self_attention.embed_dim
#
            #    # Replace the original projection layers with the LoRA layers
            #    setattr(enc_layer.self_attention, f"{char}_proj",
            #            LoRA(
            #                getattr(enc_layer.self_attention, f"{char}_proj"),
            #                rank=rank,
            #                dim=dim,
            #                initialize=True,
            #                init_type=lora_init
            #            )
            #            )
                

        #self_attention_layers = [
        #    m for n, m in self.ast_model.named_modules() if n.endswith('attn')]
        #
        #for layer in self_attention_layers:
        #    # replace attention layer with DropkeyAttention, keep other parameters the same
        #    original_state_dict = layer.state_dict()
        #    dropkeyattention = DropkeyAttention(dim = layer.qkv.in_features, num_heads=layer.num_heads, qkv_bias=layer.qkv.bias is not None, qk_scale=layer.scale, attn_drop=layer.attn_drop.p, proj_drop=layer.proj_drop.p, mask_ratio=self.p_drop)
        #    dropkeyattention.load_state_dict(original_state_dict)   
        #    self_attention_layers[i] = dropkeyattention                

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_mode == BatchMode.REPEAT:
            x = x.repeat_interleave(self.n_members, dim=0)

        out = self.ast_model(x)

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

        return dict(self.ast_model.named_parameters())

    def set_params(self, model_state_dict: Dict[str, Tensor]):
        """
        Set the parameters of the model based on a model state dict

        Parameters
        ----------
        model_state_dict : Dict[str, Tensor]
            The model state dict to set the parameters from
        """
        for name, param in model_state_dict.items():
            if name in self.ast_model.state_dict():
                # If the parameter exists in the second model, load it
                self.ast_model.state_dict()[name].copy_(param)

        #raise NotImplementedError("This method is not implemented yet.")

        ## Set the parameters of the model
        #for key, value in model_state_dict.items():
        #    # Set for encoder layers
        #    if 'blocks' in key:
        #        # Get the layer and projection
        #        layer_name = "_".join(key.split("_")[:3])
        #        layer = self.ast_model.blocks.layers.__getattr__(layer_name)
        #        proj_name = "_".join(key.split("_")[3:5])
        #        proj = layer.self_attention.__getattr__(proj_name)
#
        #        # Set the parameter
        #        proj.__setattr__("_".join(key.split("_")[5:]), value)
        #    # Set for head (legacy without ensemble head)
        #    elif 'heads.head' in key:
        #        # Catch legacy model with trainable base_model parameters
        #        if 'base_model' in key:
        #            continue
        #        # Set the parameter for the head
        #        self.ast_model.mlp_head.__setattr__(key.split(".")[-1], value)
        #    # Set for ensemble head
        #    elif 'head' in key:
        #        self.ast_head.model.heads.head.params.__setattr__(key.split("_")[-1], value)
        #    # Set for other trainable parameters
        #    else:
        #        self.vit_model.__setattr__(key, value)
