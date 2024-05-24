#!/usr/bin/env python

"""
implements the cross validation pipeline
"""

### IMPORTS ###
# Built-in imports
import sys
import json
from typing import Dict
import shutil
from pathlib import Path

# Lib imports
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# Custom imports
from experiment_settings.settings import get_settings
from model_training_evaluation.train_evaluate import train_evaluate_ensemble
import const
from utils_GPU import DEVICE
from utils_uncertainty import _ECELoss


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"

            
### FUNCTIONS ###
def cross_validation(settings: Dict, file_path: str, delete_files: bool=True):
    """
    Function to perform model training and evaluation with cross validation.

    Parameters
    ----------
    settings : Dict
        Dictionary containing the settings for the model training and evaluation.
    file_path : str
        Path to the settings file.
    delete_files : bool
        Boolean indicating whether to delete the temporary helper files after the evaluation.
    """

    # Construct dataframes for temporary storage of results
    train_loss_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    val_loss_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    accuracy_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    f1_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    precision_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    recall_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    ece_dataframe = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])
    distance_predicted_distributions = pd.DataFrame(columns = ["epoch" + str(i) for i in range(1, settings["training_settings"]["max_epochs"]+1)])

    # Store the model name across all folds (needed for global aggregation)
    model_name = settings["data_settings"]["result_file_name"]
    settings["data_settings"]["model_name"] = model_name

    # Train and evaluate the model for the first fold
    train_evaluate_ensemble(settings)

    # Set the path to the results
    path_results = const.STATS_DIR.joinpath("{}_stats_cross_validation.csv".format(settings["data_settings"]["result_file_name"])) 
        
    # read the results
    results = pd.read_csv(path_results)

    # add column accuracy as row to another dataframe, whith epoch1 to max epoch as columns
    train_loss_dataframe.loc[len(train_loss_dataframe)] = results["train_loss"].to_list()
    val_loss_dataframe.loc[len(val_loss_dataframe)] = results["val_loss"].to_list()
    accuracy_dataframe.loc[len(accuracy_dataframe)] = results["accuracy"].to_list()
    f1_dataframe.loc[len(f1_dataframe)] = results["f1"].to_list()
    precision_dataframe.loc[len(precision_dataframe)] = results["precision"].to_list()
    recall_dataframe.loc[len(recall_dataframe)] = results["recall"].to_list()
    ece_dataframe.loc[len(ece_dataframe)] = results["ece"].to_list()
    distance_predicted_distributions.loc[len(distance_predicted_distributions)] = results["distance_predicted_distributions"].to_list()

    # Save the results filename
    results_filenames = [settings["data_settings"]["result_file_name"]]

    # Delete temporary files
    if delete_files:
        path_results_csv = const.STATS_DIR.joinpath("{}_stats.csv".format(settings["data_settings"]["result_file_name"])) 
        path_results.unlink()
        path_results_csv.unlink()

    for i in range(2, settings["training_settings"]["cross_validation_nr_folds"]+1):
        # Read json file with cross validation settings
        data = json.load(open(const.SETTINGS_DIR.joinpath(file_path)))
        # Change the fold number
        data["training_settings"]["cross_validation_fold"] = i

        # Update settings path
        n_members = settings["model_settings"]["nr_members"]
        file_path_pathlib = Path(file_path)
        settings_path_new = const.SETTINGS_DIR.joinpath(
            file_path_pathlib.stem + "_members" + str(n_members) + "_fold" + str(i) + file_path_pathlib.suffix)

        print("Settings path new: ", settings_path_new)
        
        # Save the new settings and overwrite the old ones
        with open(settings_path_new, 'w') as f:
            json.dump(data, f, indent=4)

        # Train and evaluate the model for the new fold
        settings_new = get_settings(path = settings_path_new, ensemble_type= sys.argv[2], nr_members = int(sys.argv[3]))
        settings_new["data_settings"]["model_name"] = model_name
        train_evaluate_ensemble(settings_new)

        # Set the path to the results
        path_results = const.STATS_DIR.joinpath("{}_stats_cross_validation.csv".format(settings_new["data_settings"]["result_file_name"])) 
        
        # Read the results
        results = pd.read_csv(path_results)

        # Add column accuracy as row to another dataframe, whith epoch1 to max epoch as columns
        train_loss_dataframe.loc[len(train_loss_dataframe)] = results["train_loss"].to_list()
        val_loss_dataframe.loc[len(val_loss_dataframe)] = results["val_loss"].to_list()
        accuracy_dataframe.loc[len(accuracy_dataframe)] = results["accuracy"].to_list()
        f1_dataframe.loc[len(f1_dataframe)] = results["f1"].to_list()
        precision_dataframe.loc[len(precision_dataframe)] = results["precision"].to_list()
        recall_dataframe.loc[len(recall_dataframe)] = results["recall"].to_list()
        ece_dataframe.loc[len(ece_dataframe)] = results["ece"].to_list()
        distance_predicted_distributions.loc[len(distance_predicted_distributions)] = results["distance_predicted_distributions"].to_list()
        results_filenames.append(settings_new["data_settings"]["result_file_name"])

        # Delete temporary files
        if delete_files:
            path_results_csv = const.STATS_DIR.joinpath("{}_stats.csv".format(settings_new["data_settings"]["result_file_name"])) 
            settings_path_new.unlink()
            path_results.unlink()
            path_results_csv.unlink()

    # calculate mean for each column
    mean_train_loss_per_epoch = train_loss_dataframe.mean()
    mean_val_loss_per_epoch = val_loss_dataframe.mean()
    mean_accuracy_per_epoch = accuracy_dataframe.mean()
    mean_f1_per_epoch = f1_dataframe.mean()
    mean_precision_per_epoch = precision_dataframe.mean()
    mean_recall_per_epoch = recall_dataframe.mean()
    mean_ece_per_epoch = ece_dataframe.mean()
    mean_distance_predicted_distributions = distance_predicted_distributions.mean()

    # Get the epoch with the highest mean accuracy
    best_epoch = int(mean_accuracy_per_epoch.idxmax().replace("epoch", ""))

    print("Best epoch: ", best_epoch)

    # Get the stats for the best epoch
    final_train_loss = mean_train_loss_per_epoch.iloc[best_epoch-1]
    final_val_loss = mean_val_loss_per_epoch.iloc[best_epoch-1]
    final_accuracy = mean_accuracy_per_epoch.iloc[best_epoch-1]
    final_f1 = mean_f1_per_epoch.iloc[best_epoch-1]
    final_precision = mean_precision_per_epoch.iloc[best_epoch-1]
    final_recall = mean_recall_per_epoch.iloc[best_epoch-1]
    final_ece = mean_ece_per_epoch.iloc[best_epoch-1]
    final_distance_predicted_distributions = mean_distance_predicted_distributions.iloc[best_epoch-1]

    # Log results
    print("Final train loss: ", final_train_loss)
    print("Final val loss: ", final_val_loss)
    print("Final accuracy: ", final_accuracy)
    print("Final f1: ", final_f1)
    print("Final precision: ", final_precision)
    print("Final recall: ", final_recall)
    print("Final ece: ", final_ece)
    print("Final distance predicted distributions: ", final_distance_predicted_distributions)

    # Save the results
    path_results = const.STATS_DIR.joinpath("{}_stats.csv".format(settings["data_settings"]["result_file_name"]))
    results = pd.DataFrame({"train_loss": final_train_loss, "val_loss": final_val_loss, "accuracy": final_accuracy, "f1": final_f1, "precision": final_precision, "recall": final_recall, "ece": final_ece, "distance_predicted_distributions": final_distance_predicted_distributions}, index=[0])
    results.to_csv(path_results, index=False)

    # Perform global aggregation
    global_aggregate(settings, best_epoch, delete_files)


def global_aggregate(settings: Dict, best_epoch: int, delete_files: bool=True):
    """
    Aggregate the results of the cross validation for the best epoch by recalculating the stats
    for all samples in the dataset based on the predictions of the best epoch.

    Parameters
    ----------
    settings : Dict
        Dictionary containing the settings for the model training and evaluation.
    best_epoch : int
        The epoch with the highest mean accuracy.
    delete_files : bool
        Boolean indicating whether to delete the temporary helper files after the evaluation.
    """

    # Get the model name
    model_name = settings["data_settings"]["result_file_name"]
    # Construct the temporary directory path
    tmp_dir = const.STORAGE_DIR.joinpath("tmp", model_name)
    # Load the labels
    label_file = tmp_dir.joinpath(f"{model_name}_labels.npy")
    labels = np.load(label_file)
    # Load the predictions (highest probability class)
    prediction_file = tmp_dir.joinpath(f"{model_name}_predictions_epoch{best_epoch}.npy")
    predictions = np.load(prediction_file)
    # Load the logits
    logits_prediction_file = tmp_dir.joinpath(f"{model_name}_logits_epoch{best_epoch}.npy")
    logits_prediction = np.load(logits_prediction_file)

    # Define the ECE criterion
    if DEVICE == 'cuda':
        #ece_criterion = _ECELoss(multi_label=settings["data_settings"]["multi_label"]).cuda()
        ece_criterion = _ECELoss().cuda()
    else:
        #ece_criterion = _ECELoss(multi_label=settings["data_settings"]["multi_label"])
        ece_criterion = _ECELoss()

    # Construct the reliability diagram filename
    reliability_diagram_name = "reliability_diagram_globalcv_{}".format(settings['data_settings']['result_file_name'])
    # Calculate the ECE and plot the reliability diagram
    ece, accs, confs, accuracy, avg_conf = ece_criterion.forward(torch.tensor(logits_prediction).to(DEVICE),
                                                                 torch.tensor(labels).to(DEVICE),
                                                                 plot=True,
                                                                 file_name=reliability_diagram_name)

    # Calculate the f1, precision and recall
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    # Construct file path for the globalcv stats
    path_results = const.STATS_DIR.joinpath(f"{settings['data_settings']['result_file_name']}_globalcv_stats.csv")
    stats_path = const.STATS_DIR.joinpath(path_results)

    # Create stats string
    header_string = ("train_loss,val_loss,accuracy,f1,precision,recall,ece,disagreement,"
                     "distance_predicted_distributions\n")
    stats_string = (f"{np.nan},{np.nan},{accuracy},{f1},"
                    f"{precision},{recall},{ece[0]},{np.nan},{np.nan}\n")

    # Write stats to file
    with open(stats_path, "w") as stats_file:
        stats_file.write(header_string)
        stats_file.write(stats_string)

    # Delete temporary files
    if delete_files:
        shutil.rmtree(tmp_dir)
