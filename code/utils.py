"""Functions for data import."""
import os

import pandas as pd
import scipy.io as sio


def get_dict_raw_data(path, IDs, save_wt_path=False):
    """Returning a train and test dataframe consisting of the pre-processed MRI data and labels for all subjects.

    Some Description
    """
    raw_data_train = {"label_id": [], "task_id": [], "mat": []}
    raw_data_test = {"label_id": [], "task_id": [], "mat": []}

    # do a dataframe where in the column data we have either the data or the path to the data (less heavy)
    dataframe_train = pd.DataFrame(columns=["label_id", "task_id", "mat"])
    dataframe_test = pd.DataFrame(columns=["label_id", "task_id", "mat"])

    for subject_id in IDs:
        # path = '/media/miplab-nas2/Data3/Hamid/SSBCAPs/HCP100/'

        folder_id = path + str(subject_id)
        # LOAD THE 20 files .MAT if the name contain '400'
        files = [f for f in os.listdir(folder_id) if "400" in f]
        for _nb, file in enumerate(files):
            # for training dataset
            if "RL" in file:
                # if save_wt_path == True, then we save the path to the mat file
                if save_wt_path is True:
                    mat_t = folder_id + "/" + file
                else:
                    mat_t = sio.loadmat(folder_id + "/" + file)

                label_id_t = file.split(".")[0]
                task_id_t = file.split("_")[1]
                # complete dict with label_id and taks and mat
                raw_data_train["label_id"].append(label_id_t)
                raw_data_train["task_id"].append(task_id_t)
                raw_data_train["mat"].append(mat_t["v"])

            # for test dataset
            if "LR" in file:
                if save_wt_path is True:
                    mat_t = folder_id + "/" + file
                else:
                    mat_t = sio.loadmat(folder_id + "/" + file)

                label_id_t = file.split(".")[0]
                task_id_t = file.split("_")[1]
                # complete dict with label_id and taks and mat
                raw_data_test["label_id"].append(label_id_t)
                raw_data_test["task_id"].append(task_id_t)
                raw_data_test["mat"].append(mat_t["v"])

    dataframe_train["label_id"] = raw_data_train["label_id"]
    dataframe_train["task_id"] = raw_data_train["task_id"]
    dataframe_train["mat"] = raw_data_train["mat"]

    dataframe_test["label_id"] = raw_data_test["label_id"]
    dataframe_test["task_id"] = raw_data_test["task_id"]
    dataframe_test["mat"] = raw_data_test["mat"]
    # return raw_data_train, raw_data_test
    return dataframe_train, dataframe_test
