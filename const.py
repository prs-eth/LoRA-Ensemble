### IMPORTS ###
# Built-in imports
import os
from pathlib import Path

# Lib imports

# Custom imports

### AUTHORSHIP INFORMATION ###
__author__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__email__ = ["hamich@ethz.ch", "dmuehelema@ethz.ch"]
__credits__ = ["Michelle Halbheer", "Dominik Mühlematter"]
__version__ = "0.0.1"
__status__ = "Development"


### FLAGS ###
cluster = False

### DIRECTORIES ###
ROOT_DIR = Path(__file__).resolve().parents[0]  # Root directory of the project
PARENT_DIR = ROOT_DIR.parents[0]  # Parent directory of the project
MODELS_DIR = ROOT_DIR.joinpath("models")  # Directory for the model definitions
DATA_DIR = ROOT_DIR.joinpath("data")  # Directory for the data
LOGS_DIR = ROOT_DIR.joinpath("runs")  # Directory for the logs (tensorboard)
SETTINGS_DIR = ROOT_DIR.joinpath("experiment_settings")  # Directory for the experiment settings
STORAGE_DIR = ROOT_DIR.joinpath("storage")  # Directory for the storage of the models, plots, and stats, as well as slurm logs
# Directory for the storage of the models
if cluster:
    # On cluster cannot save models to working directory due to space limitations
    MODEL_STORAGE_DIR = Path(os.environ["SCRATCH"]).joinpath("models")
else:
    MODEL_STORAGE_DIR = STORAGE_DIR.joinpath("models")
PLOT_DIR = STORAGE_DIR.joinpath("plots")  # Directory for the plots
STATS_DIR = STORAGE_DIR.joinpath("stats")  # Directory for the stats
TABLES_DIR = STORAGE_DIR.joinpath("tables")  # Directory for the tables
LOGS_DIR = STORAGE_DIR.joinpath("logs")  # Directory for the slurm logs