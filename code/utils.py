"""Functions for data import."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

###-------------------------------------------------------------------------------------------------------------------
#         general utils
###-------------------------------------------------------------------------------------------------------------------


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
    if val_frac > 0:
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
    else:
        train_subjects = dataset_dataframe
        test_subjects = None
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
            train_subjects = pd.concat([train_subjects, subject_tasks])
            test_subjects = test_subjects.drop(subject_tasks.index)
            print(
                f"Moved {len(subject_tasks)} tasks from subject {subject} to the train set"
            )
    return train_subjects, test_subjects


###-------------------------------------------------------------------------------------------------------------------
#         utils for model interpretation
###-------------------------------------------------------------------------------------------------------------------


def get_atlas_mapping(
    atlas_path="Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv",
):
    """Returns the atlas mapping."""
    parcellations_df = pd.read_csv(atlas_path)
    parcellations_df["Hemisphere"] = np.where(
        parcellations_df["ROI Name"].str.contains("LH"), "LH", "RH"
    )
    # network mapping
    mapping = {
        "Vis": "VN",
        "SomMot": "SMN",
        "DorsAttn": "DAN",
        "SalVentAttn": "VAN",
        "Limbic": "LN",
        "Cont": "FPN",
        "Default": "DMN",
    }
    mapping_order = {}
    for i, k in enumerate(mapping.keys()):
        mapping_order[mapping[k]] = i
    parcellations_df["Network"] = ""
    for network in mapping:
        parcellations_df["Network"] = np.where(
            parcellations_df["ROI Name"].str.contains(network),
            mapping[network],
            parcellations_df["Network"],
        )

    parcellations_df.sort_values(
        by=["Network", "Hemisphere"],
        key=lambda x: x.map(mapping_order),
        inplace=True,
    )
    parcellations_df.reset_index(inplace=True)
    return parcellations_df


def get_mapping():
    """Returns the mapping between the network names and the network ids."""
    return {
        "Vis": "VN",
        "SomMot": "SMN",
        "DorsAttn": "DAN",
        "SalVentAttn": "VAN",
        "Limbic": "LN",
        "Cont": "FPN",
        "Default": "DMN",
    }


def get_network_ids_for_plots(mat_path="Schaefer400Yeo7.mat"):
    """Returns the indices of the networks for plotting."""
    mapping = get_mapping()
    indices_mat = sio.loadmat(mat_path)
    indices_array_LH = indices_mat["Schaefer400Yeo7"][0][0][0][0][:200]
    indices_array_RH = indices_mat["Schaefer400Yeo7"][0][0][0][0][200:]
    # for each unique value, get the first and last index of that value
    indices_LH = []
    indices_RH = []
    for i in np.unique(indices_array_LH):
        indices_LH.append(
            [
                np.where(indices_array_LH == i)[0][0],
                np.where(indices_array_LH == i)[0][-1],
            ]
        )
    for i in np.unique(indices_array_RH):
        indices_RH.append(
            [
                np.where(indices_array_RH == i)[0][0],
                np.where(indices_array_RH == i)[0][-1],
            ]
        )
    indices_LH = np.array(indices_LH)
    indices_LH[1:, 0] = indices_LH[1:, 0] - 1
    indices_RH = np.array(indices_RH)
    indices_RH = np.array(indices_RH) + 200
    indices_RH[1:, 0] = indices_RH[1:, 0] - 1
    networks_ids_for_plot_LH = {}
    for i, network in enumerate(indices_LH):
        networks_ids_for_plot_LH[list(mapping.values())[i] + "_LH"] = network
    networks_ids_for_plot_RH = {}
    for i, network in enumerate(indices_RH):
        networks_ids_for_plot_RH[list(mapping.values())[i] + "_RH"] = network
    return {**networks_ids_for_plot_LH, **networks_ids_for_plot_RH}


def reorder_network_ids_for_plots(networks_ids_for_plot):
    """Reorders the networks ids for plots."""
    mapping = get_mapping()
    networks_ids_for_plot_reordered = {}
    for i, _network in enumerate(mapping.keys()):
        networks_ids_for_plot_reordered[
            list(mapping.values())[i] + "_LH"
        ] = networks_ids_for_plot[list(mapping.values())[i] + "_LH"]
        networks_ids_for_plot_reordered[
            list(mapping.values())[i] + "_RH"
        ] = networks_ids_for_plot[list(mapping.values())[i] + "_RH"]
    return networks_ids_for_plot_reordered


def make_networks_ids_contiguous(networks_ids_for_plot):
    """Makes the networks ids contiguous.""."""
    networks_ids_for_plot_remapped = {}
    networks_ids_for_plot = reorder_network_ids_for_plots(
        networks_ids_for_plot
    )
    for network, indices in networks_ids_for_plot.items():
        if network == "VN_LH":
            previous = indices
            networks_ids_for_plot_remapped[network] = indices
            continue
        start, finish = indices
        finish - start
        i = previous[1]
        networks_ids_for_plot_remapped[network] = [i, i + finish - start]
        previous = [i, i + finish - start]
    # networks_ids_for_plot = networks_ids_for_plot_remapped
    return networks_ids_for_plot_remapped


def network_mean(matrix, network_ids=None):
    """Computes the mean of the matrix for each network."""
    if network_ids is None:
        network_ids = get_network_ids_for_plots()
    mean_matrix = np.zeros_like(matrix)
    for _k1, v1 in network_ids.items():
        for _k2, v2 in network_ids.items():
            mean_matrix[v1[0] : v1[1] + 1, v2[0] : v2[1] + 1] = (
                matrix[v1[0] : v1[1] + 1, v2[0] : v2[1] + 1].flatten().mean()
            )
    return mean_matrix


def move_networks_to_adjacent_rows(matrix):
    """Moves networks to adjacent positions in the matrix for each hemisphere."""
    matrix_reordered = np.zeros(matrix.shape)
    networks_ids_for_plot = get_network_ids_for_plots()
    current_index = 0
    total_crop_length = 0
    total_matrix_length = 0
    for _k, v in networks_ids_for_plot.items():
        start, finish = v
        crop = matrix[start:finish, :]
        # print(f"Moving {start}:{finish} to {current_index}:{current_index+crop.shape[0]}")
        # print(f"Length of crop: {crop.shape[0]}")
        total_crop_length += crop.shape[0]
        matrix_reordered[
            current_index : current_index + crop.shape[0], :
        ] = crop
        # print(f"Length of matrix_reordered: {matrix_reordered[current_index:current_index+crop.shape[0],:].shape[0]}")
        total_matrix_length += matrix_reordered[
            current_index : current_index + crop.shape[0], :
        ].shape[0]
        current_index += crop.shape[0]
        # print(f"Total crop length: {total_crop_length}")
        # print(f"Total matrix_reordered length: {total_matrix_length}")
    return matrix_reordered


def move_networks_to_adjacent_columns(matrix):
    """Moves networks to adjacent positions in the matrix for each hemisphere."""
    matrix_reordered = np.zeros(matrix.shape)
    networks_ids_for_plot = get_network_ids_for_plots()
    current_index = 0
    total_crop_length = 0
    total_matrix_length = 0
    for _k, v in networks_ids_for_plot.items():
        start, finish = v
        crop = matrix[:, start:finish]
        # print(f"Moving {start}:{finish} to {current_index}:{current_index+crop.shape[1]}")
        # print(f"Length of crop: {crop.shape[0]}")
        total_crop_length += crop.shape[1]
        matrix_reordered[
            :, current_index : current_index + crop.shape[1]
        ] = crop
        # print(f"Length of matrix_reordered: {matrix_reordered[current_index:current_index+crop.shape[0],:].shape[0]}")
        total_matrix_length += matrix_reordered[
            :, current_index : current_index + crop.shape[1]
        ].shape[1]
        current_index += crop.shape[1]
        # print(f"Total crop length: {total_crop_length}")
        # print(f"Total matrix_reordered length: {total_matrix_length}")
    return matrix_reordered


def move_networks_to_adjacent(matrix):
    """Moves networks to adjacent positions in the matrix for each hemisphere."""
    matrix = move_networks_to_adjacent_rows(matrix)
    return move_networks_to_adjacent_columns(matrix)


def save_interpretability_array_to_mat(
    interpretability_array, name="interpretability_array.mat", mean_axis=1
):
    """Saving the interpretability array to a .mat file."""
    hemispheres = ["LH", "RH"]
    full_data_dict = {}
    for hemisphere in hemispheres:
        if hemisphere == "LH":
            attributions_H = interpretability_array[:, :, :200, :200]
        else:
            attributions_H = interpretability_array[:, :, 200:, 200:]
        task_labels = [
            "REST1",
            "EMOTION",
            "GAMBLING",
            "LANGUAGE",
            "MOTOR",
            "RELATIONAL",
            "SOCIAL",
            "WM",
        ]
        data_dict = {"subject": attributions_H[0]}
        for i, task in enumerate(task_labels):
            data_dict[task] = attributions_H[i + 1]
        for k, v in data_dict.items():
            data_dict[k] = np.abs(v).mean(axis=0).mean(axis=mean_axis)
            print(data_dict[k].shape)
        full_data_dict[hemisphere] = data_dict
    sio.savemat(name, full_data_dict)
