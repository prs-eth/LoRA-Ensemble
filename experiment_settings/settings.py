#!/usr/bin/env python

"""
Functions for reading and interpreting json files for achieving hyperparameters
"""

### IMPORTS ###
# Built-in imports
import json
import os

# Lib imports
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd

# Custom imports
from model_training_evaluation.scheduler import Cosine_Schedule, Constant_Schedule, Linear_Schedule
import const
from data.utils_data import download_CIFAR10, calculate_pixel_mean_and_variance, download_CIFAR100, \
    class_weights_inverse_num_of_samples, class_weights_effective_num_of_samples
from data.project_data_loader import CIFAR10, CIFAR100, HAM10000
from data.data_augmentation import create_data_augmentation
from data.audio_datalaoder import AudiosetDataset
from models.vision_transformer import VisionTransformer
from models.AST import ASTLoRAEnsemble, ExplicitASTEnsemble, ASTModel
from models.lora_ensemble import LoRAEnsemble
from models.mc_dropout import MCDropoutEnsemble, ASTMCDropoutEnsemble
from models.lora import Init_Weight
from models.vision_transformer import VisionTransformerEnsemble, Init_Head
from utils_GPU import DEVICE
import utils


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### FUNCTIONS ###
def get_dataloader(data: dict, train: bool, transform: transforms.Compose) -> None:
    """
    Function to get the dataloader for the dataset. The function also downloads the dataset if it is not already present.

    Parameters
    ----------
    data : dict
        Dictionary containing the data settings.
    train : bool
        Boolean indicating whether the dataset is for training or evaluation.
    transform : transforms.Compose
        Transformations to be applied to the dataset.
    """

    if data["data_settings"]["data_set"] == "ESC50":
        audio_conf = {'num_mel_bins': 128, 'target_length': 512, 'freqm': 24, 'timem': 96, 'mixup': 0,
                      'dataset': 'esc50', 'mode': 'train', 'mean': -6.6268077, 'std': 5.358466,
                  'noise':False}
        val_audio_conf = {'num_mel_bins': 128, 'target_length':512, 'freqm': 0, 'timem': 0, 'mixup': 0,
                          'dataset': 'esc50', 'mode': 'evaluation', 'mean': -6.6268077, 'std': 5.358466, 'noise': False}

        if train:
            dataset = AudiosetDataset(const.DATA_DIR.joinpath('datasets/ESC50/datafiles/esc_train_data_{}.json'
                                                              .format(data["training_settings"]["cross_validation_fold"])),
                                      label_csv=const.DATA_DIR.joinpath('datasets/ESC50/esc_class_labels_indices.csv'),
                                      audio_conf=audio_conf)
        else:
            dataset = AudiosetDataset(const.DATA_DIR.joinpath('datasets/ESC50/datafiles/esc_eval_data_{}.json'
                                                              .format(data["training_settings"]["cross_validation_fold"])),
                                      label_csv=const.DATA_DIR.joinpath('datasets/ESC50/esc_class_labels_indices.csv'),
                                      audio_conf=val_audio_conf)

    elif data["data_settings"]["data_set"] == "CIFAR10":

        # get path
        sub_dir = const.DATA_DIR.joinpath('datasets/CIFAR10/')

        # download dataset
        download_CIFAR10(path=sub_dir)

        # get paths to training files
        if train:
            files = [sub_dir.joinpath("cifar-10-batches-py", f'{i}') for i in data["data_settings"]["training_files"]]

        # get paths to evaluation files
        else:
            files = [sub_dir.joinpath("cifar-10-batches-py", f'{i}') for i in data["data_settings"]["evaluation_files"]]
        dataset = CIFAR10(files, transform=transform)
    # Implement CIFAR100
    elif data["data_settings"]["data_set"] == "CIFAR100":
        # get path
        sub_dir = const.DATA_DIR.joinpath('datasets/CIFAR100/')

        # download dataset
        download_CIFAR100(path=sub_dir)

        # get paths to training files
        if train:
            files = [sub_dir.joinpath("cifar-100-python", f'{i}') for i in data["data_settings"]["training_files"]]
        else:
            files = [sub_dir.joinpath("cifar-100-python", f'{i}') for i in data["data_settings"]["evaluation_files"]]
        dataset = CIFAR100(files, transform=transform)

    elif data["data_settings"]["data_set"] == "HAM10000":
        # path to dataset
        sub_dir = const.DATA_DIR.joinpath('datasets/HAM10000')

        # load labels
        labels = pd.read_csv(sub_dir.joinpath('HAM10000_metadata.csv'))[["image_id", "dx"]]

        # create numeric labels
        labels["labels"] = labels.groupby("dx", sort=False).ngroup()

        # create train test split
        if "training_files" not in data["data_settings"] or "evaluation_files" not in data["data_settings"]:
            train_files, test_files = train_test_split(labels, test_size=0.2, stratify=labels["labels"],
                                                       random_state=42)
            data["data_settings"]["training_files"] = train_files
            data["data_settings"]["evaluation_files"] = test_files

        # get paths to training files and test files
        if train:
            files = data["data_settings"]["training_files"]
        else:
            files = data["data_settings"]["evaluation_files"]

        dataset = HAM10000(files, sub_dir, transform=transform)

    # Case of other datasets
    else:
        raise ValueError("Dataset not supported or recognized")

    # get training dataloader and add it to the data dictionary
    if train:
        data["training_settings"]["training_dataloader"] = (
            DataLoader(dataset, batch_size=data["training_settings"]["training_batch_size"],
                       shuffle=data["data_settings"]["shuffle"], num_workers=data["data_settings"]["num_workers"]))
    # get evaluation dataloader and add it to the data dictionary
    else:
        data["evaluation_settings"]["evaluation_dataloader"] = (
            DataLoader(dataset, batch_size=data["evaluation_settings"]["evaluation_batch_size"],
                       shuffle=False, num_workers=data["data_settings"]["num_workers"]))


def get_data_augmentation(data: dict, train) -> None:
    """
    Adds the data augmentation to the data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    train : bool
        Boolean indicating whether the data augmentation is for training or evaluation.
    """

    flip_image = False
    rotate_image = False
    standardize = False
    mean_pixel = None
    std_pixel = None

    # If standardize is part of the data augmentation
    if "standardize" not in data["evaluation_settings"]["evalution_augmentation"]:
        standardize = False
    elif ("standardize" in data["evaluation_settings"]["evalution_augmentation"]
            or "standardize" in data["training_settings"]["training_augmentation"]):
        standardize = True

        # If mean and standard deviation are part of the data settings
        if "channel_mean" in data["data_settings"] and "channel_std" in data["data_settings"]:
            mean_pixel = torch.tensor(data["data_settings"]["channel_mean"])
            std_pixel = torch.tensor(data["data_settings"]["channel_std"])

        # If not, calculate the mean and standard deviation
        else:
            training_augmentation = data["training_settings"]["training_augmentation"]
            get_dataloader(data, True, create_data_augmentation(image_size=data["data_settings"]["original_size"]))
            mean_pixel, std_pixel = calculate_pixel_mean_and_variance(
                data["training_settings"]["training_dataloader"],
                image_with=data["data_settings"]["original_size"][0],
                image_height=data["data_settings"]["original_size"][1])
            data["training_settings"]["training_augmentation"] = training_augmentation
            data["data_settings"]["channel_mean"] = mean_pixel.tolist()
            data["data_settings"]["channel_std"] = std_pixel.tolist()
            print("Channel mean: ", data["data_settings"]["channel_mean"])
            print("Channel std: ", data["data_settings"]["channel_std"])

    # Data augmentation for training
    if train:
        # If flip is part of the data augmentation
        if "flip" in data["training_settings"]["training_augmentation"]:
            flip_image = True

        # If rotate is part of the data augmentation
        if "rotate" in data["training_settings"]["training_augmentation"]:
            rotate_image = True

        # Create the data augmentation and add it to the data dictionary
        data["training_settings"]["training_augmentation"] = create_data_augmentation(
            data["data_settings"]["input_size"], flip_image, rotate_image, standardize, mean_pixel, std_pixel)

    # Data augmentation for evaluation
    else:

        # Create the data augmentation and add it to the data dictionary
        data["evaluation_settings"]["evalution_augmentation"] = create_data_augmentation(
            data["data_settings"]["input_size"], False, False, standardize, mean_pixel, std_pixel)


def get_optimizer(data: dict) -> None:
    """
    Function to get the optimizer for the training. The optimizer is added to the settings dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    """

    # If Adam is the optimizer
    if data["training_settings"]["optimizer"] == "Adam":
        # Initalize the optimizer and specify the settings
        lr = data["training_settings"]["learning_rate"]
        betas = data["training_settings"]["Adam_betas"]
        weight_decay = data["training_settings"]["weight_decay"]
        data["training_settings"]["optimizer"] = torch.optim.Adam([torch.tensor([])], lr=lr, betas=betas,
                                                                  weight_decay=weight_decay)

    # If SGD is the optimizer
    elif data["training_settings"]["optimizer"] == "SGD":
        # Initialize the optimizer and specify the settings
        lr = data["training_settings"]["learning_rate"]
        mom = data["training_settings"]["SGD_momentum"]
        nesterov = data["training_settings"]["SGD_nesterov"]
        weight_decay = data["training_settings"]["weight_decay"]
        data["training_settings"]["optimizer"] = torch.optim.SGD([torch.tensor([])], lr=lr, momentum=mom,
                                                                 nesterov=nesterov, weight_decay=weight_decay)
    # If AdamW is the optimizer
    elif data["training_settings"]["optimizer"] == "AdamW":
        # Initialize the optimizer and specify the settings
        lr = data["training_settings"]["learning_rate"]
        betas = data["training_settings"]["Adam_betas"]
        weight_decay = data["training_settings"]["weight_decay"]
        data["training_settings"]["optimizer"] = torch.optim.Adam([torch.tensor([])], lr=lr, betas=betas,
                                                                  weight_decay=weight_decay)
    # If anything else is given as optimizer
    else:
        raise ValueError("Optimizer not implemented or recognized")


def get_learning_rate_schedule(data: dict) -> None:
    """
    Function to get the learning rate schedule for the training.
    The learning rate schedule is added to the settings dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    """

    # Add the learning rate schedule to the settings dictionary
    if data["training_settings"]["lr_schedule"] == "cosine":
        data["training_settings"]["lr_schedule"] = Cosine_Schedule(data["training_settings"]["optimizer"],
                                                                   data["training_settings"]["steps_lr_warmup"],
                                                                   data["training_settings"]["max_steps"])
        data["training_settings"]["lr_schedule_name"] = "cosine"
    elif data["training_settings"]["lr_schedule"] == "constant":
        data["training_settings"]["lr_schedule"] = Constant_Schedule(data["training_settings"]["optimizer"],
                                                                     data["training_settings"]["steps_lr_warmup"])
        data["training_settings"]["lr_schedule_name"] = "constant"
    elif data["training_settings"]["lr_schedule"] == "linear":
        data["training_settings"]["lr_schedule"] = Linear_Schedule(data["training_settings"]["optimizer"],
                                                                   data["training_settings"]["steps_lr_warmup"],
                                                                   data["training_settings"]["max_steps"])
        data["training_settings"]["lr_schedule_name"] = "linear"

    elif data["training_settings"]["lr_schedule"] == "epoch_step":
        data["training_settings"]["lr_schedule"] = torch.optim.lr_scheduler.MultiStepLR(data["training_settings"]["optimizer"],
                                                                                  list(range(data["training_settings"]["first_epoch_step"], 1000, 1)),
                                                                                  gamma=data["training_settings"]["epoch_step_gamma"])
        data["training_settings"]["lr_schedule_name"] = "epoch_step"
    else:
        raise ValueError("Learning rate schedule not implemented or recognized")


def get_loss(data: dict) -> None:
    """
    Function to get the loss for the training. The loss is added to the settings dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    """

    # If the class weights are uniform
    if data["training_settings"]["class_weights"] == "uniform":
        data["training_settings"]["class_weights"] = torch.full((data["data_settings"]["num_classes"],),
                                                                1 / data["data_settings"]["num_classes"])

    # If the class weights are inverse number of samples
    elif data["training_settings"]["class_weights"] == "INS":

        # calculate samples per class
        samples_per_class = data["training_settings"]["training_dataloader"].dataset.labels["labels"].value_counts()

        # order by index
        samples_per_class = samples_per_class.sort_index().values

        # calculate class weights
        data["training_settings"]["class_weights"] = torch.tensor(
            class_weights_inverse_num_of_samples(data["data_settings"]["num_classes"], samples_per_class,
                                                 power=1)).float()

    # if the class weights are inverse to the square root of the number of samples
    elif data["training_settings"]["class_weights"] == "ISNS":

        # calculate samples per class
        samples_per_class = data["training_settings"]["training_dataloader"].dataset.labels["labels"].value_counts()

        # order by index
        samples_per_class = samples_per_class.sort_index().values

        # calculate class weights
        data["training_settings"]["class_weights"] = torch.tensor(
            class_weights_inverse_num_of_samples(data["data_settings"]["num_classes"], samples_per_class,
                                                 power=0.5)).float()

    elif data["training_settings"]["class_weights"] == "ENS":

        # calculate samples per class
        samples_per_class = data["training_settings"]["training_dataloader"].dataset.labels["labels"].value_counts()

        # order by index
        samples_per_class = samples_per_class.sort_index().values

        # calculate class weights
        data["training_settings"]["class_weights"] = torch.tensor(
            class_weights_effective_num_of_samples(data["data_settings"]["num_classes"], samples_per_class,
                                                   beta=data["training_settings"]["ENS_beta"])).float()

    else:
        data["training_settings"]["class_weights"] = torch.tensor(data["training_settings"]["class_weights"])

    # Print class weights
    print("Class weights: ", data["training_settings"]["class_weights"])

    # Define loss function
    if data["training_settings"]["loss"] == "cross_entropy":
        data["training_settings"]["loss"] = torch.nn.CrossEntropyLoss(weight=data["training_settings"]["class_weights"])
    elif data["training_settings"]["loss"] == "BCE":
        data["training_settings"]["loss"] = torch.nn.BCEWithLogitsLoss(weight=data["training_settings"]["class_weights"])
    else:
        raise ValueError("Loss not implemented or recognized")


def read_interpret_json(path: str) -> dict:
    """
    Function to read and interpret the json file.
    Adds implementations for data augmentation, optimizer, learning rate schedule and loss.

    Parameters
    ----------
    path : str
        Path to the json file.

    Returns
    -------
    data : dict
        Dictionary containing the settings.
    """

    # Read the json file
    data = json.load(open(path))

    # Implement data augmentation and dataloader for training 
    if data["training_settings"]["training"] is True:
        # Implement data augmentation for training 
        get_data_augmentation(data, train=True)
        get_dataloader(data, train=True, transform=data["training_settings"]["training_augmentation"])

    # Implement data augmentation and dataloader for evaluation
    if data["evaluation_settings"]["evaluation"] is True:
        # Implement data augmentation for evaluation 
        get_data_augmentation(data, train=False)
        get_dataloader(data, train=False, transform=data["evaluation_settings"]["evalution_augmentation"])

    # Implement optimizer
    get_optimizer(data)

    # Implement learning rate schedule
    get_learning_rate_schedule(data)

    # Implement loss
    get_loss(data)

    return data


def get_model(data: dict, ensemble_type: str) -> torch.nn.Module:
    """
    Function to get the model for the training and evaluation. The model is added to the settings dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the settings.
    ensemble_type : str
        Type of the ensemble model.

    Returns
    -------
    ensemble_model : torch.nn.Module
        The ensemble model.
    """

    # Define the perturbation scale and the maximum perturbation layer if not defined
    if "perturb_scale" not in data["model_settings"]:
        data["model_settings"]["perturb_scale"] = None
    if "max_perturb_layer" not in data["model_settings"]:
        data["model_settings"]["max_perturb_layer"] = None

    # Initialize dict for head weight initialization settings
    head_settings = {}

    # Set the initialization method for the head
    if "head_init" not in data["model_settings"]:
        init_head = Init_Head.DEFAULT
        head_settings = None
    elif data["model_settings"]["head_init"] == "gaussian":
        init_head = Init_Head.NORMAL
        head_settings["std"] = data["model_settings"]["gaussian_std"]
        head_settings["mean"] = data["model_settings"]["gaussian_mean"]
    elif data["model_settings"]["head_init"] == "kaiming":
        init_head = Init_Head.KAIMING_UNIFORM
        head_settings["a_squared"] = data["model_settings"]["kaiming_a_squared"]
    elif data["model_settings"]["head_init"] == "xavier_uniform":
        init_head = Init_Head.XAVIER_UNIFORM
        head_settings["gain"] = data["model_settings"]["xavier_gain"]
    else:
        init_head = Init_Head.DEFAULT
        head_settings = None

    # Implement Explicit Ensemble
    if ensemble_type == "Deep_Ensemble":
        ensemble_model = VisionTransformerEnsemble(
            n_members=data["model_settings"]["nr_members"],
            n_classes=data["data_settings"]["num_classes"],
            config=data["model_settings"]["ViT_config"],
            patch_size=data["model_settings"]["ViT_patch_size"],
            pretrained=True,
            perturb_scale=data["model_settings"]["perturb_scale"],
            max_perturb_layer=data["model_settings"]["max_perturb_layer"],
            init_head=init_head,
            init_settings=head_settings
        )

        ensemble_params = []
        for model in ensemble_model.vit_models:
            ensemble_params.extend(model.parameters())

        # Pass the model params
        data["model_settings"]["model_params"] = ensemble_params

        # Add parameters to the optimizer
        data["training_settings"]["optimizer"].param_groups[0]["params"] = ensemble_params

        # Calculate number of parameters
        n_params = 0
        for param in ensemble_params:
            n_params += param.numel()

    # Implement LoRA-Ensemble
    elif ensemble_type == "LoRA_Former":

        # Initialize dict for LoRA weight initialization settings
        init_settings = {}

        # Set the initialization method for the LoRA module
        if data["LoRA_settings"]["weight_init"] == "gaussian":
            lora_init = Init_Weight.NORMAL
            init_settings["std"] = data["LoRA_settings"]["gaussian_std"]
            init_settings["mean"] = data["LoRA_settings"]["gaussian_mean"]

        elif data["LoRA_settings"]["weight_init"] == "kaiming":
            lora_init = Init_Weight.KAIMING_UNIFORM
            init_settings["a_squared"] = data["LoRA_settings"]["kaiming_a_squared"]

        elif data["LoRA_settings"]["weight_init"] == "xavier_uniform":
            lora_init = Init_Weight.XAVIER_UNIFORM
            init_settings["gain"] = data["LoRA_settings"]["xavier_gain"]

        else:
            lora_init = Init_Weight.DEFAULT
            init_settings = None

        # Create the LoRA-Ensemble model
        ensemble_model = LoRAEnsemble(VisionTransformer(data["data_settings"]["num_classes"],
                                                        data["model_settings"]["ViT_config"],
                                                        data["model_settings"]["ViT_patch_size"]).to(DEVICE),
                                                        data["LoRA_settings"]["rank"],
                                                        data["model_settings"]["nr_members"],
                                                        lora_init=lora_init,
                                                        init_settings=init_settings,
                                                        init_head=init_head,
                                                        head_settings=head_settings
                                      )
        # Gather the model parameters that need training 
        ensemble_params = ensemble_model.gather_params()

        # Pass the model params
        data["model_settings"]["model_params"] = ensemble_params

        # Add parameters to the optimizer
        data["training_settings"]["optimizer"].param_groups[0]["params"] = ensemble_params.values()

        # Calculate number of parameters
        n_params = 0
        for param in ensemble_params.values():
            n_params += param.numel()

    # Implement AST-LoRA-Ensemble
    elif ensemble_type == "AST_Former":
        
        # Initialize dict for LoRA weight initialization settings
        init_settings = {}

        # Set the initialization method for the LoRA module
        if data["LoRA_settings"]["weight_init"] == "gaussian":
            lora_init = Init_Weight.NORMAL
            init_settings["std"] = data["LoRA_settings"]["gaussian_std"]
            init_settings["mean"] = data["LoRA_settings"]["gaussian_mean"]

        elif data["LoRA_settings"]["weight_init"] == "kaiming":
            lora_init = Init_Weight.KAIMING_UNIFORM
            init_settings["a_squared"] = data["LoRA_settings"]["kaiming_a_squared"]

        elif data["LoRA_settings"]["weight_init"] == "xavier_uniform":
            lora_init = Init_Weight.XAVIER_UNIFORM
            init_settings["gain"] = data["LoRA_settings"]["xavier_gain"]

        else:
            lora_init = Init_Weight.DEFAULT
            init_settings = None

        if "ensemble_layer_norm" not in data["LoRA_settings"]:
            data["LoRA_settings"]["ensemble_layer_norm"] = False

        if "chunk_size" not in data["LoRA_settings"]:
            data["LoRA_settings"]["chunk_size"] = None

        # Create the LoRA-Ensemble model
        ensemble_model = ASTLoRAEnsemble(label_dim=data["data_settings"]["num_classes"],
                                         fstride=data["data_settings"]["fstride"],
                                         tstride=data["data_settings"]["tstride"], input_fdim=128,
                                         input_tdim=data["data_settings"]["audio_length"],
                                         imagenet_pretrain=data["model_settings"]["imagenet_pretraining"],
                                         audioset_pretrain=data["model_settings"]["audioset_pretraining"],
                                         model_size=data["model_settings"]["DeiT_version"],
                                         n_members=data["model_settings"]["nr_members"],
                                         lora_init=lora_init, init_settings=init_settings, init_head=init_head,
                                         head_settings=head_settings, rank=data["LoRA_settings"]["rank"],
                                         train_patch_embed=data["LoRA_settings"]["train_conv"],
                                         train_pos_embed=data["LoRA_settings"]["train_pos"],
                                         ensemble_layer_norm=data["LoRA_settings"]["ensemble_layer_norm"],
                                         chunk_size=data["LoRA_settings"]["chunk_size"])

        # Gather the model parameters that need training
        ensemble_params = ensemble_model.gather_params()

        # Pass the model params
        data["model_settings"]["model_params"] = ensemble_params

        # Add parameters to the optimizer
        data["training_settings"]["optimizer"].param_groups[0]["params"] = ensemble_params.values()

        # Calculate number of parameters
        n_params = 0
        for param in ensemble_params.values():
            n_params += param.numel()

    # Implement Explicit AST-Ensemble
    elif ensemble_type == "Explicit_AST":
        ensemble_model = ExplicitASTEnsemble(
            n_members=data["model_settings"]["nr_members"],
            n_classes=data["data_settings"]["num_classes"],
            config=data["model_settings"]["ViT_config"],
            patch_size=data["model_settings"]["ViT_patch_size"],
            pretrained=True,
            perturb_scale=data["model_settings"]["perturb_scale"],
            max_perturb_layer=data["model_settings"]["max_perturb_layer"],
            init_head=init_head,
            init_settings=head_settings,
            fstride=data["data_settings"]["fstride"],
            tstride=data["data_settings"]["tstride"],
            input_fdim= 128,
            input_tdim=data["data_settings"]["audio_length"],
            imagenet_pretrain=data["model_settings"]["imagenet_pretraining"],
            audioset_pretrain=data["model_settings"]["audioset_pretraining"],
            model_size=data["model_settings"]["DeiT_version"]
        )

        ensemble_params = []
        for model in ensemble_model.ast_models:
            ensemble_params.extend(model.parameters())

        # Pass the model params
        data["model_settings"]["model_params"] = ensemble_params

        # Add parameters to the optimizer
        data["training_settings"]["optimizer"].param_groups[0]["params"] = ensemble_params

        # Calculate number of parameters
        n_params = 0
        for param in ensemble_params:
            n_params += param.numel()
    # Implement MC Dropout
    elif ensemble_type == "MCDropout":
        # Initialize dict for head weight initialization settings
        head_settings = {}

        # Set the initialization method for the head
        if "head_init" not in data["model_settings"]:
            init_head = Init_Head.DEFAULT
            head_settings = None
        elif data["model_settings"]["head_init"] == "gaussian":
            init_head = Init_Head.NORMAL
            head_settings["std"] = data["model_settings"]["gaussian_std"]
            head_settings["mean"] = data["model_settings"]["gaussian_mean"]
        elif data["model_settings"]["head_init"] == "kaiming":
            init_head = Init_Head.KAIMING_UNIFORM
            head_settings["a_squared"] = data["model_settings"]["kaiming_a_squared"]
        elif data["model_settings"]["head_init"] == "xavier_uniform":
            init_head = Init_Head.XAVIER_UNIFORM
            head_settings["gain"] = data["model_settings"]["xavier_gain"]
        else:
            init_head = Init_Head.DEFAULT
            head_settings = None

        ensemble_model = MCDropoutEnsemble(
            VisionTransformer(
                data["data_settings"]["num_classes"],
                data["model_settings"]["ViT_config"],
                data["model_settings"]["ViT_patch_size"],
                custom_attention=False
            ).to(DEVICE),
            data["model_settings"]["nr_members"],
            p_drop=data["MCDropout_settings"]["p_drop"],
            init_head=init_head,
            head_settings=head_settings
        )
        # Gather the model parameters that need training
        ensemble_params = ensemble_model.gather_params()

        # Pass the model params
        data["model_settings"]["model_params"] = ensemble_params

        # Add parameters to the optimizer
        data["training_settings"]["optimizer"].param_groups[0]["params"] = ensemble_params.values()

        # Calculate number of parameters
        n_params = 0
        for param in ensemble_params.values():
            n_params += param.numel()

    # Implement MC Dropout
    elif ensemble_type == "ASTMCDropout":
        # Initialize dict for head weight initialization settings
        head_settings = {}

        # Set the initialization method for the head
        if "head_init" not in data["model_settings"]:
            init_head = Init_Head.DEFAULT
            head_settings = None
        elif data["model_settings"]["head_init"] == "gaussian":
            init_head = Init_Head.NORMAL
            head_settings["std"] = data["model_settings"]["gaussian_std"]
            head_settings["mean"] = data["model_settings"]["gaussian_mean"]
        elif data["model_settings"]["head_init"] == "kaiming":
            init_head = Init_Head.KAIMING_UNIFORM
            head_settings["a_squared"] = data["model_settings"]["kaiming_a_squared"]
        elif data["model_settings"]["head_init"] == "xavier_uniform":
            init_head = Init_Head.XAVIER_UNIFORM
            head_settings["gain"] = data["model_settings"]["xavier_gain"]
        else:
            init_head = Init_Head.DEFAULT
            head_settings = None

        ensemble_model = ASTMCDropoutEnsemble(
            ASTModel(
            label_dim=data["data_settings"]["num_classes"],
            fstride=data["data_settings"]["fstride"],
            tstride=data["data_settings"]["tstride"],
            input_fdim= 128,
            input_tdim=data["data_settings"]["audio_length"],
            imagenet_pretrain=data["model_settings"]["imagenet_pretraining"],
            audioset_pretrain=data["model_settings"]["audioset_pretraining"],
            model_size=data["model_settings"]["DeiT_version"]
            ).to(DEVICE),
            data["model_settings"]["nr_members"],
            p_drop=data["MCDropout_settings"]["p_drop"],
            init_head=init_head,
            head_settings=head_settings
        )
        # Gather the model parameters that need training
        ensemble_params = ensemble_model.gather_params()

        # Pass the model params
        data["model_settings"]["model_params"] = ensemble_params

        # Add parameters to the optimizer
        data["training_settings"]["optimizer"].param_groups[0]["params"] = ensemble_params.values()

        # Calculate number of parameters
        n_params = 0
        for param in ensemble_params.values():
            n_params += param.numel()

        print(ensemble_model)

    else:
        raise ValueError("Ensemble type not implemented or recognized")

    # Add the model name to the settings dictionary
    data["model_settings"]["model_name"] = \
        (f"{ensemble_type}_ViT_{data['model_settings']['ViT_config']}_patch_{data['model_settings']['ViT_patch_size']}"
         f"_members_{data['model_settings']['nr_members']}")

    # Print the model info and the number of trainable parameters
    print("Ensemble model: {} ViT-{}-{} with {} members".format(ensemble_type, data["model_settings"]["ViT_config"],
                                                                data["model_settings"]["ViT_patch_size"],
                                                                data["model_settings"]["nr_members"]))
    print(f"Number of trainable parameters: {n_params}")

    return ensemble_model


def get_settings(path: str, ensemble_type: str, nr_members: int) -> dict:
    """
    Function to get the settings for the training and evaluation. The settings are added to the settings dictionary.

    Parameters
    ----------
    path : str
        Path to the json file.
    ensemble_type : str
        Type of the ensemble model.
    nr_members : int
        Number of ensemble members.

    Returns
    -------
    data : dict
        Dictionary containing the settings.
    """

    # Process the json file
    data = read_interpret_json(path)

    # Set the random seed
    seed = data["training_settings"]["random_seed"]
    utils.set_seed(seed)

    # Add the number of ensemble members to the settings dictionary
    data["model_settings"]["nr_members"] = nr_members

    # Add the ensemble model to the settings dictionary
    data["model_settings"]["ensemble_type"] = ensemble_type
    data["model_settings"]["model"] = get_model(data, ensemble_type)

    # extract name of json from path
    data["model_settings"]["json_file"] = os.path.basename(path).split(".")[0]

    # Generate file name for saving results and models
    data["data_settings"]["json_file_name"] = os.path.basename(path).split(".")[0]
    data["data_settings"]["result_file_name"] = \
        (f"{ensemble_type}_ViT_{data['model_settings']['ViT_config']}_{data['model_settings']['ViT_patch_size']}_"
         f"{data['model_settings']['nr_members']}_members_{data['data_settings']['json_file_name']}")
    data["data_settings"]["result_file_name"] = data["data_settings"]["result_file_name"].replace("-", "_")

    if "timing" not in data["evaluation_settings"]:
        data["evaluation_settings"]["timing"] = False

    # Add the summary writer to the settings dictionary
    if data["data_settings"]["tensorboard"] is True:
        data["data_settings"]["tensorboard_writer"] = SummaryWriter(
            const.LOGS_DIR.joinpath('runs/{}'.format(data["data_settings"]["result_file_name"])))

    return data
