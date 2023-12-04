"""Runs nested cross validation."""
#
###-------------------------------------------------------------------------------------------------------------------
#         imports
###-------------------------------------------------------------------------------------------------------------------
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from cross_validation import training_loop_and_cross_validation
from sklearn.preprocessing import LabelEncoder
from utils import get_df_raw_data

## Data path ##
DATA_PATH = (Path.cwd().parent / "DATA").resolve()
print(f"Data path: {DATA_PATH}")
DATA_PATH = str(DATA_PATH)
# DATA_PATH = "/media/miplab-nas2/Data3/Hamid/SSBCAPs/HCP100"

###-------------------------------------------------------------------------------------------------------------------
#         subject ID list
###-------------------------------------------------------------------------------------------------------------------

IDs = [
    100307,
    117122,
    131722,
    153025,
    211720,
    100408,
    118528,
    133019,
    154734,
    212318,
    101107,
    118730,
    133928,
    156637,
    214423,
    101309,
    118932,
    135225,
    159340,
    221319,
    101915,
    120111,
    135932,
    160123,
    239944,
    103111,
    122317,
    136833,
    161731,
    245333,
    103414,
    122620,
    138534,
    162733,
    280739,
    103818,
    123117,
    139637,
    163129,
    298051,
    105014,
    123925,
    140925,
    176542,
    366446,
    105115,
    124422,
    144832,
    178950,
    397760,
    106016,
    125525,
    146432,
    188347,
    414229,
    108828,
    126325,
    147737,
    189450,
    499566,
    110411,
    127630,
    148335,
    190031,
    654754,
    111312,
    127933,
    148840,
    192540,
    672756,
    111716,
    128127,
    149337,
    196750,
    751348,
    113619,
    128632,
    149539,
    198451,
    756055,
    113922,
    129028,
    149741,
    199655,
    792564,
    114419,
    130013,
    151223,
    201111,
    856766,
    115320,
    130316,
    151526,
    208226,
    857263,
]

#
###-------------------------------------------------------------------------------------------------------------------
#         joining train and test dataframes from all subjects
###-------------------------------------------------------------------------------------------------------------------

data_df_train, data_df_test = get_df_raw_data(DATA_PATH, IDs[:])

NUM_SUBJECTS = len(data_df_train["subject_id"].unique())
print(f"Number of subjects: {NUM_SUBJECTS}")
###-------------------------------------------------------------------------------------------------------------------
#         label encoding
###-------------------------------------------------------------------------------------------------------------------

# label encoding
enc_labels = LabelEncoder()
enc_tasks = LabelEncoder()

enc_labels.fit(data_df_train["subject_id"].tolist())
enc_tasks.fit(data_df_train["task"].tolist())

enc_train_label_encodings = enc_labels.transform(
    data_df_train["subject_id"].tolist()
)
enc_train_task_encodings = enc_tasks.transform(data_df_train["task"].tolist())


enc_test_label_encodings = enc_labels.transform(
    data_df_test["subject_id"].tolist()
)
enc_test_task_encodings = enc_tasks.transform(data_df_test["task"].tolist())

data_df_train["enc_subject_id"] = enc_train_label_encodings
data_df_train["enc_task_id"] = enc_train_task_encodings
data_df_test["enc_subject_id"] = enc_test_label_encodings
data_df_test["enc_task_id"] = enc_test_task_encodings

###-------------------------------------------------------------------------------------------------------------------
#         cross validation
###-------------------------------------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

criterion = nn.CrossEntropyLoss()

config = {
    # general
    "epochs": 30,
    "batch_size": 32,
    # optimizer
    "lambda_si": 0.5,
    "lambda_td": 0.5,
    "k_folds": 2,
    "num_subjects": NUM_SUBJECTS,
    "d_model_input": 400,
}

###-------------------------------------------------------------------------------------------------------------------
#         hyperparameter combinations
###-------------------------------------------------------------------------------------------------------------------

learning_rate = [
    # 1e-5, 
    1e-4, 
    # 1e-3, 
    # 1e-2
    ]
dropout = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9
    ]
intermediate_size = [None, [500, 250], [250], [1000]]
layer_norm = [True]


# Get all possible configuration combination of the parameters above
all_model_combinations = list(
    itertools.product(learning_rate, dropout, intermediate_size, layer_norm)
)
criterion = nn.CrossEntropyLoss()

(
    average_error_per_linear_split_model,
    average_error_per_linear_shared_model,
    model_parameters,
) = training_loop_and_cross_validation(
    data_df_train, criterion, device, all_model_combinations, config
)

min_mean_loss_linear_split_model_indice = np.argmin(
    average_error_per_linear_split_model[:, 0]
)
min_mean_loss_linear_split_model = average_error_per_linear_split_model[
    min_mean_loss_linear_split_model_indice
]
optimal_parameters_linear_split_model = model_parameters[
    min_mean_loss_linear_split_model_indice
]

min_mean_loss_linear_shared_model_indice = np.argmin(
    average_error_per_linear_shared_model[:, 0]
)
min_mean_loss_linear_shared_model = average_error_per_linear_shared_model[
    min_mean_loss_linear_shared_model_indice
]
optimal_parameters_linear_shared_model = model_parameters[
    min_mean_loss_linear_shared_model_indice
]

with Path("cv_run.txt").open("w") as f:
    f.write("Linear Split Model\n")
    f.write("Average loss across folds for all combinations:\n")
    f.write(str(average_error_per_linear_split_model) + "\n")
    f.write("Average loss across folds of best performing model:\n")
    f.write(str(min_mean_loss_linear_split_model) + "\n")
    f.write(
        "Optimal Hyperparameters (learning_rate, dropout, intermediate_size, layer_norm):\n"
    )
    f.write(str(optimal_parameters_linear_split_model) + "\n")

    f.write("Linear Shared Model\n")
    f.write("Average loss across folds for all combinations:\n")
    f.write(str(average_error_per_linear_shared_model) + "\n")
    f.write("Average loss across folds of best performing model:\n")
    f.write(str(min_mean_loss_linear_shared_model) + "\n")
    f.write(
        "Optimal Hyperparameters (learning_rate, dropout, intermediate_size, layer_norm):\n"
    )
    f.write(str(optimal_parameters_linear_shared_model) + "\n")
    f.close()
