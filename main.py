#!/usr/bin/env python

"""
implements the training and testing pipeline for CIFAR10
"""

### IMPORTS ###
# Built-in imports
import sys

# Lib imports

# Custom imports
from experiment_settings.settings import get_settings
from model_training_evaluation.train_evaluate import train_evaluate_ensemble
import const
import utils
from models.model_loader import load_model
import json
import pandas as pd
from model_training_evaluation.cross_validation import cross_validation

### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### EXECUTE ###
if __name__ == "__main__":
    """
    Function for training and/or evaluating ViT ensemble models.
    """
    
    if len(sys.argv) > 1:
        # Iterate over the command-line arguments
        print("Command-line arguments:")
        print("1. Name of the json file containing the experiment settings: ", sys.argv[1])
        print("2. Type of the ensemble model: ", sys.argv[2])
        print("3. Number of the ensemble members: ", sys.argv[3])
        print("")

        # Set the path to the experiment settings
        settings_path = const.SETTINGS_DIR.joinpath(sys.argv[1])

        # Get the experiment settings
        settings = get_settings(path = settings_path, ensemble_type= sys.argv[2], nr_members = int(sys.argv[3]))

        # If a saved model is provided, load the model
        if len(sys.argv) == 5:
            settings["model_settings"]["model"] = load_model(sys.argv[4], settings["model_settings"]["model"])
            settings["training_settings"]["training"] = False

        # If cross-validation is enabled, perform cross-validation
        if "cross_validation" in settings["training_settings"].keys() and settings["training_settings"]["cross_validation"] == True:
            cross_validation(settings, sys.argv[1])

        # If training is enabled, train/evaluate the model
        else:
            # Train and evaluate the ensemble model
            train_evaluate_ensemble(settings)

    else:
        print("Not all command-line arguments are provided.")
        print("Please provide the following command-line arguments:")
        print("1. Name of the json file containing the experiment settings.")
        print("2. Type of the ensemble model ('Deep-Ensemble' , 'LoRA-Former').")
        print("3. Number of the ensemble members.")
        print("")
        print("Example: python main.py CIFAR10_settings_experiment1.json LoRA-Former 2")
        # python main.py ESC50_test_dominik_experiment1.json AST_Former 1
        # python main.py ESC50_settingsPaper_experiment1.json Explicit_AST 1 
        # python main.py ESC50_settingsPaperLoRA_experiment1.json AST_Former 1
        # python main.py CIFAR100_settings_experiment12_copy.json LoRA_Former 2
        # python main.py ESC50_settings_MCDropout1.json ASTMCDropout 2
