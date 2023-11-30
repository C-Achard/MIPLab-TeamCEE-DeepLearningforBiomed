"""Nested Cross Validation Training."""
###-------------------------------------------------------------------------------------------------------------------
#         imports
###-------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from models import MRIAttention
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from training import balanced_data_shuffle_cv, training_loop


def training_loop_and_cross_validation(
    data_df_train, criterion, device, model_parameter_combinations, config
):
    """Runs cross validation on the train set.

    returns cross validation loss of every model using different hyperparameters to select best ones
    """
    ###-------------------------------------------------------------------------------------------------------------------
    #          K-fold crossvalidation
    ###----------------------------------------------------------------------------------------------------------------
    CV = KFold(config["k_folds"], shuffle=True)

    mean_total_loss = []
    model_parameters = []

    for _k, (train_index, test_index) in enumerate(
        CV.split(
            data_df_train["mat"],
            data_df_train["enc_subject_id"].to_numpy(),
            data_df_train["enc_task_id"].to_numpy(),
        )
    ):
        # erstelllen von dingens
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

        # train all combinatorial models
        for i, properties in enumerate(model_parameter_combinations):
            print(
                "Model Iteration: "
                + str(i)
                + " / "
                + str(len(model_parameter_combinations))
            )

            # Get model configurations
            dropout = properties[0]
            attention_dropout = properties[1]
            num_heads = properties[2]
            learning_rate = properties[3]

            model = MRIAttention(
                output_size_tasks=9,
                output_size_subjects=config["num_subjects"],
                input_size=config["d_model_input"],
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

            history = training_loop(
                config["epochs"],
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                config,
            )

            mean_total_loss.append(np.mean(history["val-loss_total"][-10:]))
            model_parameters.append(model_parameter_combinations)

    return (
        mean_total_loss,
        model_parameters,
    )
