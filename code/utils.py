"""Functions for data import."""
from pathlib import Path

import pandas as pd
import scipy.io as sio


def get_df_raw_data(path, IDs, save_wt_path=False):
    """Returning a train and test dataframe consisting of the pre-processed MRI data and labels for all subjects.

    Some Description
    """
    raw_data_train = {"subject_id": [], "task": [], "mat": []}
    raw_data_test = {"subject_id": [], "task": [], "mat": []}

    # do a dataframe where in the column data we have either the data or the path to the data (less heavy)
    dataframe_train = pd.DataFrame(columns=["subject_id", "task", "mat"])
    dataframe_test = pd.DataFrame(columns=["subject_id", "task", "mat"])

    for subject_id in IDs:
        path = Path(path)
        folder_id = path / str(subject_id)
        # LOAD THE 20 files .MAT if the name contain '400'
        # files = [f for f in folder_id.iterdir() if "400" in f]
        files = [f for f in folder_id.iterdir() if "400" in f.name]
        for _nb, file in enumerate(files):
            # for training dataset
            file = str(file).split("/")[-1]
            if "RL" in file:
                task_id_t = file.split("_")[1]
                if task_id_t == "REST2":
                    continue
                # if save_wt_path == True, then we save the path to the mat file
                if save_wt_path is True:
                    mat_t = str(folder_id / file)
                else:
                    mat_t = sio.loadmat(str(folder_id / file))
                # label_id_t = file.split(".")[0]
                label_id_t = Path(file).name.split(".")[0]

                # complete dict with label_id and taks and mat
                raw_data_train["subject_id"].append(label_id_t)
                raw_data_train["task"].append(task_id_t)
                raw_data_train["mat"].append(mat_t["v"])

            # for test dataset
            if "LR" in file:
                task_id_t = file.split("_")[1]
                if task_id_t == "REST2":
                    continue

                if save_wt_path is True:
                    mat_t = str(folder_id / file)
                else:
                    mat_t = sio.loadmat(str(folder_id / file))

                label_id_t = Path(file).name.split(".")[0]

                # remove NaN p-matrices containing NaN values
                if (
                    subject_id == 211720 or subject_id == 756055
                ) and task_id_t == "REST1":
                    continue

                # complete dict with label_id and taks and mat
                raw_data_test["subject_id"].append(label_id_t)
                raw_data_test["task"].append(task_id_t)
                raw_data_test["mat"].append(mat_t["v"])

    dataframe_train["subject_id"] = raw_data_train["subject_id"]
    dataframe_train["task"] = raw_data_train["task"]
    dataframe_train["mat"] = raw_data_train["mat"]

    dataframe_test["subject_id"] = raw_data_test["subject_id"]
    dataframe_test["task"] = raw_data_test["task"]
    dataframe_test["mat"] = raw_data_test["mat"]
    # return raw_data_train, raw_data_test
    return dataframe_train, dataframe_test
