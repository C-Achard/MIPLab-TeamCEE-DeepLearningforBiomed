"""Runs model training."""
#
###-------------------------------------------------------------------------------------------------------------------
#         imports
###-------------------------------------------------------------------------------------------------------------------
from pathlib import Path
import logging
import numpy as np
import torch
import torch.nn as nn

from models import MRIAttentionLinear, MRIAttention, MRICustomAttention
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from training import balanced_data_shuffle, training_loop
from utils import get_df_raw_data

## Data path ##
DATA_PATH = (Path.cwd().parent / "DATA").resolve()  # TODO : adapt to server
# DATA_PATH = Path("/media/miplab-nas2/Data3/Hamid/SSBCAPs/HCP100").resolve()
print(f"Data path: {DATA_PATH}")
DATA_PATH = str(DATA_PATH)

logging.basicConfig(level=logging.INFO)
#
# %load_ext autoreload
# %autoreload 2

#
###-------------------------------------------------------------------------------------------------------------------
#         hyperparameters
###-------------------------------------------------------------------------------------------------------------------

config = {
    # data
    "stratify": True,
    "validation_split": 0.1,
    # general
    "epochs": 100,
    "batch_size": 10,
    "lr": 1e-4,
    "use_scheduler": True,
    # model
    "d_model_input": 400,
    # "d_model_intermediate": 512,
    "d_model_task_output": 8,
    "d_model_fingerprint_output": None,  # needs to be determined from data
    # "dropout": 0,
    "attention_dropout": 0.1,
    "num_heads": 1,
    # "num_layers": 0,  # TBA?
    # optimizer
    "lambda_si": 0.5,
    "lambda_td": 0.5,
    "weight_decay": 1,
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
    data_df_train,
    val_frac=config["validation_split"],
    stratify=config["stratify"]
)
NUM_SUBJECTS = len(data_df_train["subject_id"].unique())
print(f"Number of subjects: {NUM_SUBJECTS}")
NUM_TASKS = data_df_train["task"].nunique()
print(f"Number of tasks: {NUM_TASKS}")

#
###-------------------------------------------------------------------------------------------------------------------
#         label encoding
###-------------------------------------------------------------------------------------------------------------------
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

def show_df_distribution(df):
    # print("Distribution of subjects:")
    # print(df["subject_id"].value_counts())
    # print("Distribution of tasks:")
    # print(df["task"].value_counts())
    # print("_"*20)
    print("Number of samples:", len(df))
    print("Unique subjects:", df["subject_id"].nunique())
    print("Unique tasks:", df["task"].nunique())
    print("*" * 50)
    
print("Train set:")
show_df_distribution(train_dataframe)
print("Validation set:")
show_df_distribution(valid_dataframe)
print("Test set:")
show_df_distribution(data_df_test)

print("Subjects present in train set but not in test set:")
overlap_set = set(train_dataframe["subject_id"].unique()) - set(data_df_test["subject_id"].unique())
print(overlap_set)
if len(overlap_set) != 0:
    print("WARNING: subjects present in train set but not in test set")
print("Subjects present in validation set but not in train set:")
overlap_set = set(valid_dataframe["subject_id"].unique()) - set(train_dataframe["subject_id"].unique())
print(overlap_set)
if len(overlap_set) != 0:
    print("WARNING: subjects present in validation set but not in train set")
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
    valid_dataset, batch_size=config["batch_size"], shuffle=False
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
device_list = ["cpu"] + [
    f"cuda:{i}" for i in range(torch.cuda.device_count())
]
device = device_list[-1]
# device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

## Self-Attention model ##

model = MRIAttention(
    # output_size_tasks = config["d_model_task_output"],
    output_size_tasks=NUM_TASKS,
    output_size_subjects=NUM_SUBJECTS,
    input_size=config["d_model_input"],
    attention_dropout=config["attention_dropout"],
    num_heads=config["num_heads"],
).to(device)

## Custom EGNNA model ##

# model = MRICustomAttention(
#     output_size_subjects=NUM_SUBJECTS,
#     output_size_tasks=NUM_TASKS,
#     input_size=config["d_model_input"],
#     attention_dropout=config["attention_dropout"],
# ).to(device)

# x = torch.randn(1, 400, 400)
# y = model(x.to(device))

# x_si, x_td, attn_weights
# print(y[0].size())
# print(y[1].size())
# print(y[2].size())

#
###-------------------------------------------------------------------------------------------------------------------
#         training
###-------------------------------------------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                              weight_decay=config["weight_decay"]
                              )
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.1, patience=2, verbose=True, min_lr=1e-8, cooldown=10, threshold=1e-4
# )
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.1
)

training_loop(
    config["epochs"],
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    config,
    scheduler=scheduler if config["use_scheduler"] else None,
    save_model=False,
    save_attention_weights=True,
    test_loader=test_loader
)

