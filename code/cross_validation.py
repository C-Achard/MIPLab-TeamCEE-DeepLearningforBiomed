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


def training_loop_nested_cross_validation(
    data_df_train, criterion, device, model_parameter_combinations, config
):
    """Runs nested cross validation on the train set.

    inner Loop for hyperparameter selection
    out loop for model selection
    """
    ###-------------------------------------------------------------------------------------------------------------------
    #          inner and outer K-fold crossvalidation
    ###----------------------------------------------------------------------------------------------------------------
    CV_outer = KFold(config["k_outer"], shuffle=True)
    CV_inner = KFold(config["k_inner"], shuffle=True)

    total_loss_for_optimized_model_parameters = []
    all_optimal_model_parameters = []

    for _k, (train_index, test_index) in enumerate(
        CV_outer.split(
            data_df_train["mat"],
            data_df_train["enc_subject_id"].to_numpy(),
            data_df_train["enc_task_id"].to_numpy(),
        )
    ):
        # erstelllen von dingens
        cv_outer_train_df = pd.DataFrame(
            {
                "mat": data_df_train["mat"][train_index],
                "enc_subject_id": data_df_train["enc_subject_id"][train_index],
                "enc_task_id": data_df_train["enc_task_id"][train_index],
            }
        )
        cv_outer_test_df = pd.DataFrame(
            {
                "mat": data_df_train["mat"][test_index],
                "enc_subject_id": data_df_train["enc_subject_id"][test_index],
                "enc_task_id": data_df_train["enc_task_id"][test_index],
            }
        )

        # balance shuffles
        cv_outer_train_df, cv_outer_test_df = balanced_data_shuffle_cv(
            cv_outer_train_df, cv_outer_test_df
        )

        cv_outer_train_df = cv_outer_train_df.reset_index()
        cv_outer_test_df = cv_outer_test_df.reset_index()

        # train loader
        train_dataset = TensorDataset(
            torch.Tensor(np.array(cv_outer_train_df["mat"].tolist())),
            torch.Tensor(cv_outer_train_df["enc_subject_id"].to_numpy()),
            torch.Tensor(cv_outer_train_df["enc_task_id"].to_numpy()),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )

        # test loader
        test_dataset = TensorDataset(
            torch.Tensor(np.array(cv_outer_test_df["mat"].tolist())),
            torch.Tensor(cv_outer_test_df["enc_subject_id"].to_numpy()),
            torch.Tensor(cv_outer_test_df["enc_task_id"].to_numpy()),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False
        )

        # to store all the errors for each model combination
        inner_loss_multi_attention = np.array([])

        # inner CV on current train fold
        for _k, (train_index_inner, test_index_inner) in enumerate(
            CV_inner.split(
                cv_outer_train_df["mat"],
                cv_outer_train_df["enc_subject_id"],
                cv_outer_train_df["enc_task_id"],
            )
        ):
            # erstelllen von dingens
            cv_inner_train_df = pd.DataFrame(
                {
                    "mat": cv_outer_train_df["mat"][train_index_inner],
                    "enc_subject_id": cv_outer_train_df["enc_subject_id"][
                        train_index_inner
                    ],
                    "enc_task_id": cv_outer_train_df["enc_task_id"][
                        train_index_inner
                    ],
                }
            )
            cv_inner_test_df = pd.DataFrame(
                {
                    "mat": cv_outer_train_df["mat"][test_index_inner],
                    "enc_subject_id": cv_outer_train_df["enc_subject_id"][
                        test_index_inner
                    ],
                    "enc_task_id": cv_outer_train_df["enc_task_id"][
                        test_index_inner
                    ],
                }
            )

            # balance shuffles
            cv_inner_train_df, cv_inner_test_df = balanced_data_shuffle_cv(
                cv_inner_train_df, cv_inner_test_df
            )

            # train loader
            train_dataset = TensorDataset(
                torch.Tensor(np.array(cv_inner_train_df["mat"].tolist())),
                torch.Tensor(cv_inner_train_df["enc_subject_id"].to_numpy()),
                torch.Tensor(cv_inner_train_df["enc_task_id"].to_numpy()),
            )
            train_loader_inner = DataLoader(
                train_dataset, batch_size=config["batch_size"], shuffle=True
            )

            # test loader
            test_dataset = TensorDataset(
                torch.Tensor(np.array(cv_inner_test_df["mat"].tolist())),
                torch.Tensor(cv_inner_test_df["enc_subject_id"].to_numpy()),
                torch.Tensor(cv_inner_test_df["enc_task_id"].to_numpy()),
            )
            test_loader_inner = DataLoader(
                test_dataset, batch_size=config["batch_size"], shuffle=False
            )

            # to store all errors of this inner_cv run
            inner_loss_multi_attention_current_run = []

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
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rate
                )

                history = training_loop(
                    config["epochs"],
                    model,
                    train_loader_inner,
                    test_loader_inner,
                    criterion,
                    optimizer,
                    device,
                    config,
                )

                # append test error
                inner_loss_multi_attention_current_run.append(
                    np.mean(history["val-loss_total"][-10:])
                )

            if len(inner_loss_multi_attention) == 0:
                inner_loss_multi_attention = (
                    inner_loss_multi_attention_current_run
                )
            else:
                inner_loss_multi_attention = np.vstack(
                    [
                        inner_loss_multi_attention,
                        inner_loss_multi_attention_current_run,
                    ]
                )

        # get best performing model on inner CVs
        # train all combinatorial models
        # get best performing model on the inner folds! :-)
        average_loss_per_combination = (
            np.sum(inner_loss_multi_attention, axis=0) / config["k_inner"]
        )
        optimal_average_loss_for_all_combinations = np.min(
            average_loss_per_combination
        )
        optimal_model_parameters = model_parameter_combinations[
            np.argmin(optimal_average_loss_for_all_combinations)
        ]

        # Get optimal model configurations
        dropout = optimal_model_parameters[0]
        attention_dropout = optimal_model_parameters[1]
        num_heads = optimal_model_parameters[2]
        learning_rate = optimal_model_parameters[3]

        model = MRIAttention(
            output_size_tasks=9,
            output_size_subjects=config["num_subjects"],
            input_size=config["d_model_input"],
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        history = training_loop(
            config["epochs"],
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            config,
        )

        # return
        total_loss_for_optimized_model_parameters.append(
            np.mean(history["val-loss_total"][-10:])
        )
        all_optimal_model_parameters.append(optimal_model_parameters)

        return (
            total_loss_for_optimized_model_parameters,
            all_optimal_model_parameters,
        )
    return None
