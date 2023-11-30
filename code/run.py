"""Runs model training."""
#
###-------------------------------------------------------------------------------------------------------------------
#         imports
###-------------------------------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# import sys
# sys.path.append("../code/")
from models import MRIAttention, LinearLayer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from training import balanced_data_shuffle, training_loop
from utils import get_df_raw_data

## Data path ##
#DATA_PATH = (Path.cwd().parent / "DATA").resolve()  # TODO : adapt to server
DATA_PATH = ("C:/Users/emy8/OneDrive/Documents/EPFL/Master/MA3/DeepLbiomed/Project/MIPLab-TeamCEE-DeepLearningforBiomed/DATA")
print(f"Data path: {DATA_PATH}")
#DATA_PATH = str(DATA_PATH)

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
data_df_train, data_df_test = get_df_raw_data(DATA_PATH, IDs[:])
train_dataframe, valid_dataframe = balanced_data_shuffle(
    data_df_train, test_size=0.2
)
# display(data_df_train.head(10))

#
NUM_SUBJECTS = len(data_df_train["subject_id"].unique())
print(f"Number of subjects: {NUM_SUBJECTS}")

#
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

# data_dict_train["enc_label_id"] = enc_train_label_encodings.tolist()
# data_dict_train["enc_task_id"] = enc_train_task_encodings.tolist()

# data_dict_test["enc_label_id"] = enc_test_label_encodings.tolist()
# data_dict_test["enc_task_id"] = enc_test_task_encodings.tolist()

# label encoding
enc_labels = LabelEncoder()
enc_tasks = LabelEncoder()

enc_labels.fit(data_df_train["subject_id"].tolist())
enc_tasks.fit(data_df_train["task"].tolist())

enc_train_label_encodings = enc_labels.transform(
    train_dataframe["subject_id"].tolist()
)
enc_train_task_encodings = enc_tasks.transform(
    train_dataframe["task"].tolist()
)

enc_valid_label_encodings = enc_labels.transform(
    valid_dataframe["subject_id"].tolist()
)
enc_valid_task_encodings = enc_tasks.transform(
    valid_dataframe["task"].tolist()
)

enc_test_label_encodings = enc_labels.transform(
    data_df_test["subject_id"].tolist()
)
enc_test_task_encodings = enc_tasks.transform(data_df_test["task"].tolist())

train_dataframe["enc_label_id"] = enc_train_label_encodings
train_dataframe["enc_task"] = enc_train_task_encodings
valid_dataframe["enc_label_id"] = enc_valid_label_encodings
valid_dataframe["enc_task"] = enc_valid_task_encodings
data_df_test["enc_label_id"] = enc_test_label_encodings
data_df_test["enc_task"] = enc_test_task_encodings

# enc.inverse_transform() to reverse

#
# display(data_df_train.head(10))

#
###-------------------------------------------------------------------------------------------------------------------
#         initializing dataloader objects
###-------------------------------------------------------------------------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(np.array(train_dataframe["mat"].tolist()).astype(np.float32)),
    torch.tensor(train_dataframe["enc_label_id"].to_numpy()),
    torch.tensor(train_dataframe["enc_task"].to_numpy()),
)
train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True
)

valid_dataset = TensorDataset(
    torch.tensor(np.array(valid_dataframe["mat"].tolist()).astype(np.float32)),
    torch.tensor(valid_dataframe["enc_label_id"].to_numpy()),
    torch.tensor(valid_dataframe["enc_task"].to_numpy()),
)
valid_loader = DataLoader(
    valid_dataset, batch_size=config["batch_size"], shuffle=True
)

test_dataset = TensorDataset(
    torch.tensor(np.array(data_df_test["mat"].tolist()).astype(np.float32)),
    torch.tensor(data_df_test["enc_label_id"].to_numpy()),
    torch.tensor(data_df_test["enc_task"].to_numpy()),
)
test_loader = DataLoader(
    test_dataset, batch_size=config["batch_size"], shuffle=False
)

#
###-------------------------------------------------------------------------------------------------------------------
#         initializing model
###-------------------------------------------------------------------------------------------------------------------

# list all available torch devices
# device_list = ["cpu"] + [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
# device = device_list[-1]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MRIAttention(
    # output_size_tasks = config["d_model_task_output"],
    output_size_tasks=9,
    output_size_subjects=NUM_SUBJECTS,
    input_size=config["d_model_input"],
    num_heads=config["num_heads"],
    dropout=config["dropout"],
    attention_dropout=config["attention_dropout"],
).to(device)

x = torch.randn(1, 400, 400).to(device)
y = model(x)

# x_si, x_td, attn_weights
print(y[0].size())
print(y[1].size())
print(y[2].size())

model_LL = LinearLayer(
    output_size_tasks=9,
    output_size_subjects=NUM_SUBJECTS,
    input_size=config["d_model_input"],
    #intermediate_size=[512],
    dropout=config["dropout"],
).to(device)

#
###-------------------------------------------------------------------------------------------------------------------
#         training
###-------------------------------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

training_loop(
    config["epochs"],
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    config,
)
