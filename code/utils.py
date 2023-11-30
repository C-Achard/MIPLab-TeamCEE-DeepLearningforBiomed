"""Functions for data import."""
from pathlib import Path

import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

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

def balanced_data_shuffle(dataset_dataframe, val_frac=0.2, stratify=True):
    """Shuffle and stratify the data by task, so that each task is represented equally in the train and test sets.

    Also ensures no subject is present only in the test set.
    If this is the case, half of the tasks of this subject are moved to the train set.
    """
    train_subjects, test_subjects = train_test_split(
        dataset_dataframe,
        test_size=val_frac,
        stratify=dataset_dataframe["task"] if stratify else None,
    )
    # find if subjects are present only in the test set
    test_only_subjects = test_subjects[
        ~test_subjects["subject_id"].isin(train_subjects["subject_id"])
    ]
    if len(test_only_subjects) > 0:
        print(
            f"Found {len(test_only_subjects['subject_id'].unique())} subjects present only in the test set"
        )
        # if there are subjects present only in the test set, move half of their tasks to the train set
        for subject in test_only_subjects["subject_id"].unique():
            subject_tasks = test_subjects[
                test_subjects["subject_id"] == subject
            ].sample(frac=0.5)
            train_subjects = train_subjects.append(subject_tasks)
            test_subjects = test_subjects.drop(subject_tasks.index)
            print(
                f"Moved {len(subject_tasks)} tasks from subject {subject} to the train set"
            )
    return train_subjects, test_subjects


def balanced_data_shuffle_cv(train_subjects, test_subjects):
    """Shuffle and stratify the data by task, so that each task is represented equally in the train and test sets.

    Also ensures no subject is present only in the test set.
    If this is the case, half of the tasks of this subject are moved to the train set.
    """
    # find if subjects are present only in the test set
    test_only_subjects = test_subjects[
        ~test_subjects["enc_subject_id"].isin(train_subjects["enc_subject_id"])
    ]
    if len(test_only_subjects) > 0:
        print(
            f"Found {len(test_only_subjects['enc_subject_id'].unique())} subjects present only in the test set"
        )
        # if there are subjects present only in the test set, move half of their tasks to the train set
        for subject in test_only_subjects["enc_subject_id"].unique():
            subject_tasks = test_subjects[
                test_subjects["enc_subject_id"] == subject
            ].sample(frac=0.5)
            train_subjects = train_subjects.append(subject_tasks)
            test_subjects = test_subjects.drop(subject_tasks.index)
            print(
                f"Moved {len(subject_tasks)} tasks from subject {subject} to the train set"
            )
    return train_subjects, test_subjects