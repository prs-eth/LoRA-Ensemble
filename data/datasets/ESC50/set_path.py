#!/usr/bin/env python

"""
Implements the data loaders for this project
"""

### IMPORTS ###
# Built-in imports
import json
from pathlib import Path
import os

# Lib imports

# Custom imports
import const


### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


def set_esc50_path():
    """
    Set the path of the ESC50 dataset to a realtive path to the root directory
    This way the dataset can be used on the cluster and locally without changing the path
    """
    esc_path = const.ROOT_DIR.joinpath("data", "datasets", "ESC50")

    datafiles = esc_path.joinpath("datafiles")

    for file in datafiles.iterdir():
        file_json = json.load(open(file, "r"))

        entries = file_json["data"]

        for entry in entries:
            wav = entry["wav"]
            wav_path = Path(wav)
            rel_path = wav.split("LoRA-Ensemble")[0]
            wav_new = wav_path.relative_to(rel_path)
            entry["wav"] = str(wav_new.as_posix())

        json.dump(file_json, open(file, "w"), indent=4)



def get_esc50_path(wav: str) -> Path:
    """
    Return the absolute path of a wav file in the ESC50 dataset given the relative path

    Parameters
    ----------
    wav : String
        The relative path of the wav file

    Returns
    -------
    full_path : Path
        The absolute path of the wav file
    """
    if const.cluster:
        base_path = Path(os.environ["SCRATCH"])
    else:
        base_path = const.PARENT_DIR

    wav_path = Path(wav)

    full_path = base_path.joinpath(wav_path)

    return full_path
