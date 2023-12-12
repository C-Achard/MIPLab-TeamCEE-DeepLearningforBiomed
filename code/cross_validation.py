"""Cross Validation Training - Helper functions."""
import gc

import numpy as np
import pandas as pd
import torch
from models import LinearLayer, LinearLayerShared
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from training import training_loop
from utils import balanced_data_shuffle_cv


def training_loop_and_cross_validation(
    data_df_train, criterion, device, model_parameter_combinations, config
):
    """Runs cross validation on the train set.

    returns cross validation loss of every model using different hyperparameters to select best ones
    """
    # k fold cross validation
    CV = KFold(config["k_folds"], shuffle=True)

    all_losses_linear_split_model = np.array([])
    all_losses_linear_shared_model = np.array([])

    for _k, (train_index, test_index) in enumerate(
        CV.split(
            data_df_train["mat"],
            data_df_train["enc_subject_id"].to_numpy(),
            data_df_train["enc_task_id"].to_numpy(),
        )
    ):
        CV_train_df = pd.DataFrame(
            {
                "mat": data_df_train["mat"][train_index],
                "enc_subject_id": data_df_train["enc_subject_id"][train_index],
                "enc_task_id": data_df_train["enc_task_id"][train_index],
            }
        )
        CV_test_df = pd.DataFrame(
            {
                "mat": data_df_train["mat"][test_index],
                "enc_subject_id": data_df_train["enc_subject_id"][test_index],
                "enc_task_id": data_df_train["enc_task_id"][test_index],
            }
        )

        # balance shuffles
        CV_train_df, CV_test_df = balanced_data_shuffle_cv(
            CV_train_df, CV_test_df
        )

        CV_train_df = CV_train_df.reset_index()
        CV_test_df = CV_test_df.reset_index()

        # train loader
        train_dataset = TensorDataset(
            torch.Tensor(np.array(CV_train_df["mat"].tolist())),
            torch.Tensor(CV_train_df["enc_subject_id"].to_numpy()),
            torch.Tensor(CV_train_df["enc_task_id"].to_numpy()),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )

        # val loader
        val_dataset = TensorDataset(
            torch.Tensor(np.array(CV_test_df["mat"].tolist())),
            torch.Tensor(CV_test_df["enc_subject_id"].to_numpy()),
            torch.Tensor(CV_test_df["enc_task_id"].to_numpy()),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False
        )

        current_cv_losses_linear_split_model = []
        current_cv_losses_linear_shared_model = []

        # train all combinatorial models
        for i, properties in enumerate(model_parameter_combinations):
            print(
                "Model Iteration: "
                + str(i)
                + " / "
                + str(len(model_parameter_combinations))
            )

            gc.collect()
            torch.cuda.empty_cache()

            print("Current parameters: " + str(properties))

            # Get model configurations
            learning_rate = properties[0]
            dropout = properties[1]
            intermediate_size = properties[2]
            layer_norm = properties[3]

            linear_model_split_layers = LinearLayer(
                output_size_tasks=9,
                output_size_subjects=config["num_subjects"],
                input_size=config["d_model_input"],
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm=layer_norm,
            ).to(device)

            linear_model_shared_layers = LinearLayerShared(
                output_size_tasks=9,
                output_size_subjects=config["num_subjects"],
                input_size=config["d_model_input"],
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm=layer_norm,
            ).to(device)

            optimizer_split = torch.optim.AdamW(
                linear_model_split_layers.parameters(), lr=learning_rate
            )
            optimizer_shared = torch.optim.AdamW(
                linear_model_shared_layers.parameters(), lr=learning_rate
            )

            history_split = training_loop(
                config["epochs"],
                linear_model_split_layers,
                train_loader,
                val_loader,
                criterion,
                optimizer_split,
                device,
                config,
                run_name="experiment_1",
                job_name="Linear Split"
                + " Iteration "
                + str(i)
                + " Fold "
                + str(_k),
            )

            history_shared = training_loop(
                config["epochs"],
                linear_model_shared_layers,
                train_loader,
                val_loader,
                criterion,
                optimizer_shared,
                device,
                config,
                run_name="experiment_1",
                job_name="Linear Shared"
                + " Iteration "
                + str(i)
                + " Fold "
                + str(_k),
            )

            current_cv_losses_linear_split_model.append(
                [
                    np.mean(history_split["val-loss_total"][-5:]),
                    np.mean(history_split["val-acc_si"][-5:]),
                    np.mean(history_split["val-acc_td"][-5:]),
                ]
            )

            current_cv_losses_linear_shared_model.append(
                [
                    np.mean(history_shared["val-loss_total"][-5:]),
                    np.mean(history_shared["val-acc_si"][-5:]),
                    np.mean(history_shared["val-acc_td"][-5:]),
                ]
            )

        if len(all_losses_linear_split_model) == 0:
            all_losses_linear_split_model = [
                current_cv_losses_linear_split_model
            ]
        else:
            all_losses_linear_split_model = np.vstack(
                [
                    all_losses_linear_split_model,
                    [current_cv_losses_linear_split_model],
                ]
            )

        if len(all_losses_linear_shared_model) == 0:
            all_losses_linear_shared_model = [
                current_cv_losses_linear_shared_model
            ]
        else:
            all_losses_linear_shared_model = np.vstack(
                [
                    all_losses_linear_shared_model,
                    [current_cv_losses_linear_shared_model],
                ]
            )

    average_error_per_linear_split_model = (
        np.sum(all_losses_linear_split_model, axis=0) / config["k_folds"]
    )

    average_error_per_linear_shared_model = (
        np.sum(all_losses_linear_shared_model, axis=0) / config["k_folds"]
    )

    return (
        average_error_per_linear_split_model,
        average_error_per_linear_shared_model,
        model_parameter_combinations,
    )
