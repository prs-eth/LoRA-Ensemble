# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from timm.models.layers import to_2tuple,trunc_normal_
from models.lora import ASTEnsembleLoRA, Init_Weight
from utils_GPU import DEVICE
from typing import List, Dict
from torch import nn, Tensor, vmap
import const
import enum
from torch.func import stack_module_state, functional_call
from models.vision_transformer import EnsembleHead, Init_Head
from models.lora_ensemble import BatchMode


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTLoRAEnsemble(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(self, 
                 label_dim=527, 
                 fstride=10, 
                 tstride=10, 
                 input_fdim=128, 
                 input_tdim=1024, 
                 imagenet_pretrain=True, 
                 audioset_pretrain=False, 
                 model_size='base384', 
                 verbose=True, 
                 add_LoRA=True,
                 batch_mode=BatchMode.DEFAULT,
                 rank=4,
                 lora_layers=None,
                 n_members=1,
                 init_settings=None,
                 lora_init=Init_Weight.DEFAULT,
                 lora_type="qkv",
                 init_head=Init_Weight.DEFAULT,  # Not implemented
                 head_settings=None,  # Not implemented
                 train_patch_embed=True,
                 train_pos_embed=True,
                 ensemble_layer_norm=False,
                 chunk_size=None
                 ):

        super(ASTLoRAEnsemble, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # LoRA settings
        # Set the flag whether to use LoRA
        self._add_LoRA = add_LoRA
        self.train_patch_embed = train_patch_embed
        self.train_pos_embed = train_pos_embed

        # Define the batch mode
        self.batch_mode = batch_mode  # The way batch parallelization is used through the ensemble

        # Set properties
        self.n_members = n_members  # Number of ensemble members
        self.lora_type = lora_type  # Which projections LoRA is applied to
        self.lora_layers = lora_layers
        self.rank = rank
        self.init_settings = init_settings
        self.lora_init = lora_init
        self.init_head = init_head #todo
        self.head_settings = head_settings # todo
        self.ensemble_layer_norm = ensemble_layer_norm
        self.chunk_size = chunk_size

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                if not const.DATA_DIR.joinpath(
                        'datasets/ESC50/pretrained_models/deit_base_distilled_patch16_384_ImageNet.pth').exists():
                    const.DATA_DIR.joinpath(
                        'datasets/ESC50/pretrained_models').mkdir(exist_ok=True, parents=True)
                    self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)

                    # save the model pretrained on ImageNet to local
                    torch.save(self.v.state_dict(), str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/deit_base_distilled_patch16_384_ImageNet.pth')
                else:   
                    self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                    # load the model pretrained on ImageNet from local
                    sd = torch.load(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/deit_base_distilled_patch16_384_ImageNet.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    self.v.load_state_dict(sd, strict=True)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # Add custom head
            if ensemble_layer_norm:
                # Add the ensemble head with layer norm
                self.mlp_head = EnsembleHead(self.original_embedding_dim, label_dim, n_members, layer_norm=True)
            else:
                # Add the ensemble head without layer norm
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim).to(DEVICE),
                                              EnsembleHead(self.original_embedding_dim, label_dim, n_members))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))

            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                if os.path.exists(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models') == False:
                    print("Creating directory: ", str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models')
                    os.mkdir(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models')
                wget.download(audioset_mdl_url, out=str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # Add custom head
            if ensemble_layer_norm:
                # Add the ensemble head with layer norm
                self.mlp_head = EnsembleHead(self.original_embedding_dim, label_dim, n_members, layer_norm=True)
            else:
                # Add the ensemble head without layer norm
                self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim).to(DEVICE),
                                              EnsembleHead(self.original_embedding_dim, label_dim, n_members))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

        # If custom attention is enabled set it up
        if self._add_LoRA:
            self.add_LoRA()

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def add_LoRA(self):
        # Move the model to the device
        self.v.to(DEVICE)

        # Set the layers to apply LoRA to
        if self.lora_layers is None:
            self.lora_layers = list(range(len(self.v.blocks)))

        # Freeze Vision Transformer weights
        for param in self.v.parameters():
            param.requires_grad = False

        # Apply LoRA to the specified layers
        for layer_id, enc_layer in enumerate(self.v.blocks):
            # If layer should not include LoRA, skip
            if layer_id not in self.lora_layers:
                continue

            # Extract dimensions for the projections of the layer
            dim = 768

            # Replace the original projection layers with the LoRA layers
            setattr(enc_layer.attn, "qkv",
                        ASTEnsembleLoRA(
                            getattr(enc_layer.attn, "qkv"),
                            rank=self.rank,
                            dim=768,
                            out_dim=2304,
                            n_members=self.n_members,
                            initialize=True,
                            init_type=self.lora_init,
                            init_settings=self.init_settings,
                            chunk_size=self.chunk_size
                        )
                        )
            setattr(enc_layer.attn, "proj",
                    ASTEnsembleLoRA(
                        getattr(enc_layer.attn, "proj"),
                        rank=self.rank,
                        dim=768,
                        out_dim=768,
                        n_members=self.n_members,
                        initialize=True,
                        init_type=self.lora_init,
                        init_settings=self.init_settings,
                        chunk_size=self.chunk_size
                    )
                    )

    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        
        if self.batch_mode == BatchMode.REPEAT:
            x = x.repeat_interleave(self.n_members, dim=0)

        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)

        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        out = self.mlp_head(x)
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
        for layer_id, enc_layer in enumerate(self.v.blocks):
            # If layer should not include LoRA, skip
            if layer_id not in self.lora_layers:
                continue

            proj_params = enc_layer.attn.__getattr__(f"qkv").params
            proj_params = {f"blocks_{layer_id}_qkv_{k}": v for k, v in proj_params.items()
                            if k in ["w_a.weight", "w_b.weight"]}
            params.update(proj_params)

            # Gather the parameters of the out projection
            out_proj_params = enc_layer.attn.proj.params
            out_proj_params = {f"blocks_{layer_id}_proj_{k}": v for k, v in out_proj_params.items()
                               if k in ["w_a.weight", "w_b.weight"]}
            params.update(out_proj_params)

        # Add head parameters
        if self.ensemble_layer_norm:
            for name, param in self.mlp_head.params.items():
                params.update({f"mlp_head_{name}": param})
        else:
            for name, param in self.mlp_head[0].named_parameters():
                params.update({f"mlp_head_ln.{name}": param})
            for name, param in self.mlp_head[1].params.items():
                params.update({f"mlp_head_linear.{name}": param})

        # Add parameters of the patch embed
        if self.train_patch_embed:
            for name, param in self.v.patch_embed.named_parameters():

                # make sure that the patch embedding is trainable
                param.requires_grad = True

                # Add the parameter
                params.update({f"patch_embed_{name}": param})

        # Add parameters of the positional embedding
        if self.train_pos_embed:
            for name, param in self.v.named_parameters():
                if "pos_embed" in name:

                    # make sure that the positional embedding is trainable
                    param.requires_grad = True

                    params.update({name: param})

        # Add other trainable parameters from the model
        for name, param in self.v.named_parameters():
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
            if 'blocks' in key:
                # Get the layer and projection
                self.v.blocks[int(key.split("_")[1])].attn.__getattr__(key.split("_")[2]).params["_".join(key.split("_")[3:5])] = value
 
            # Set for head 
            elif 'mlp_head' in key:
                if key.split("_")[2].split(".")[0] == "ln":
                    self.mlp_head[0].__setattr__(key.split("_")[2].split(".")[1], value)
                elif key.split("_")[2].split(".")[0] == "linear":
                    self.mlp_head[1].params[key.split("_")[2].split(".")[1]] = value
            

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                if os.path.exists(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/deit_base_distilled_patch16_384_ImageNet.pth') == False:
                    self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)

                    # save the model pretrained on ImageNet to local
                    torch.save(self.v.state_dict(), str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/deit_base_distilled_patch16_384_ImageNet.pth')
                else:   
                    self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                    # load the model pretrained on ImageNet from local
                    sd = torch.load(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/deit_base_distilled_patch16_384_ImageNet.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    self.v.load_state_dict(sd, strict=True)
                    
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                if os.path.exists(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models') == False:
                    print("Creating directory: ", str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models')
                    os.mkdir(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models')
                wget.download(audioset_mdl_url, out=str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load(str(const.DATA_DIR) + '/datasets/ESC50/pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    #@autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        
        return x
    

class ExplicitASTEnsemble(nn.Module):
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
            perturb_scale: float = None,  # Not implemented
            max_perturb_layer: int = None,  # Not implemented
            init_head: Init_Head = Init_Head.DEFAULT,  # Not implemented
            init_settings: dict = None,  # Not implemented
            fstride: int = 10,
            tstride: int = 10,
            input_fdim: int = 128,
            input_tdim: int = 1024,
            imagenet_pretrain: bool = True,
            audioset_pretrain: bool = False,
            model_size: str = 'base384'

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

        super(ExplicitASTEnsemble, self).__init__()

        self.n_members = n_members

        # Create the list of ensemble members
        self.ast_models = [ASTModel(label_dim=n_classes, 
                                    fstride=fstride, 
                                    tstride=tstride, 
                                    input_fdim=input_fdim, 
                                    input_tdim=input_tdim, 
                                    imagenet_pretrain=imagenet_pretrain, 
                                    audioset_pretrain=audioset_pretrain, 
                                    model_size=model_size, 
                                    verbose=True).to(DEVICE)
                                    for _ in range(self.n_members)]

        if perturb_scale is not None or max_perturb_layer is not None:
            raise UserWarning("Perturbation of weights is not implemented.")

        if init_head is not Init_Head.DEFAULT or init_settings is not None:
            raise UserWarning("User specified initialization of the head is not implemented.")

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

        return functional_call(self.base_model, (params, buffers), (x,))

    def forward(self, x: Tensor) -> Tensor:

        out = [model(x) for model in self.ast_models]
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
        for member in self.ast_models:
            length_model_in_list = int(len(model_state_list) / len(self.ast_models))
            model_params = model_state_list[param_index:param_index + length_model_in_list]
            count = 0
            for p in member.parameters():
                p.data = model_params[count]
                count += 1
            param_index += length_model_in_list

        #raise NotImplementedError("This method is not implemented yet.")

        ## self.params = model_state_dict
        #model_length = len(model_state_list) // self.n_members
        #for i, model in enumerate(self.vit_models):
        #    for j, param in enumerate(model.parameters()):
        #        param = model_state_list[i * model_length + j]
        #        pass
