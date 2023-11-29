"""Runs nested cross validation."""
#
###-------------------------------------------------------------------------------------------------------------------
#         imports
###-------------------------------------------------------------------------------------------------------------------
import itertools
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from cross_validation import training_loop_nested_cross_validation

# import sys
# sys.path.append("../code/")
from sklearn.preprocessing import LabelEncoder
from utils import get_df_raw_data

## Data path ##
DATA_PATH = (Path.cwd().parent / "DATA").resolve()  # TODO : adapt to server
print(f"Data path: {DATA_PATH}")
DATA_PATH = str(DATA_PATH)

#
# %load_ext autoreload
# %autoreload 2

#
###-------------------------------------------------------------------------------------------------------------------
#         hyperparameters
###-------------------------------------------------------------------------------------------------------------------

config = {
    # general
    "epochs": 100,
    "batch_size": 4,
    "lr": 1e-3,
    # model
    "d_model_input": 400,
    "d_model_intermediate": 512,
    "d_model_task_output": 8,
    "d_model_fingerprint_output": None,  # needs to be determined from data
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "num_heads": 4,
    "num_layers": 0,  # TBA?
    # optimizer
    "lambda_si": 0.5,
    "lambda_td": 0.5,
}

#
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

# data_dict_train, data_dict_test = get_dict_raw_data(DATA_PATH, IDs[0:3])
data_df_train, data_df_test = get_df_raw_data(DATA_PATH, IDs[:10])

NUM_SUBJECTS = len(data_df_train["subject_id"].unique())
print(f"Number of subjects: {NUM_SUBJECTS}")
###-------------------------------------------------------------------------------------------------------------------
#         label encoding
###-------------------------------------------------------------------------------------------------------------------

# one hot encoding

# enc_labels = OneHotEncoder(handle_unknown='ignore')
# enc_tasks = OneHotEncoder(handle_unknown='ignore')

# enc_labels.fit(data_dict_train["subject_id"].to_numpy().reshape(-1, 1))
# enc_tasks.fit(data_dict_train["task_id"].to_numpy().reshape(-1, 1))

# enc_train_label_encodings = enc_labels.transform(data_dict_train["subject_id"].to_numpy().reshape(-1, 1)).toarray()
# enc_train_task_encodings = enc_tasks.transform(data_dict_train["task_id"].to_numpy().reshape(-1, 1)).toarray()

# enc_test_label_encodings = enc_labels.transform(data_dict_test["subject_id"].to_numpy().reshape(-1, 1)).toarray()
# enc_test_task_encodings = enc_tasks.transform(data_dict_test["task_id"].to_numpy().reshape(-1, 1)).toarray()

# data_dict_train["enc_subject_id"] = enc_train_label_encodings.tolist()
# data_dict_train["enc_task_id"] = enc_train_task_encodings.tolist()

# data_dict_test["enc_subject_id"] = enc_test_label_encodings.tolist()
# data_dict_test["enc_task_id"] = enc_test_task_encodings.tolist()

# label encoding
enc_labels = LabelEncoder()
enc_tasks = LabelEncoder()

enc_labels.fit(data_df_train["subject_id"].tolist())
enc_tasks.fit(data_df_train["task_id"].tolist())

enc_train_label_encodings = enc_labels.transform(
    data_df_train["subject_id"].tolist()
)
enc_train_task_encodings = enc_tasks.transform(
    data_df_train["task_id"].tolist()
)


enc_test_label_encodings = enc_labels.transform(
    data_df_test["subject_id"].tolist()
)
enc_test_task_encodings = enc_tasks.transform(data_df_test["task_id"].tolist())

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

###-------------------------------------------------------------------------------------------------------------------
#         hyperparameter combinations
###-------------------------------------------------------------------------------------------------------------------

dropout = [0.2, 0.5]  # , 0.8]
attention_dropout = [0.2, 0.5]  # , 0.8]
num_heads = [2, 4]  # ,8]
learning_rate = [1e-3]  # , 1e-2, 1e-4]

k_inner = 2
k_outer = 2

config = {
    # general
    "epochs": 2,
    "batch_size": 4,
    # optimizer
    "lambda_si": 0.5,
    "lambda_td": 0.5,
    "k_inner": 2,
    "k_outer": 2,
    "num_subjects": NUM_SUBJECTS,
    "d_model_input": 400,
}

# Get all possible configuration combination of the parameters above
all_model_combinations = list(
    itertools.product(dropout, attention_dropout, num_heads, learning_rate)
)

criterion = nn.CrossEntropyLoss()

(
    total_loss_for_optimized_model_parameters,
    all_optimal_model_parameters,
) = training_loop_nested_cross_validation(
    data_df_train, criterion, device, all_model_combinations, config
)

counter = Counter(all_optimal_model_parameters)
print(total_loss_for_optimized_model_parameters)
print(all_optimal_model_parameters)
print(counter)
