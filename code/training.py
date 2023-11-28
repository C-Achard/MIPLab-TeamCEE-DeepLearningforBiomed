"""Functions for training and evaluating models."""
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    import wandb as wb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


def balanced_data_shuffle(dataset_dataframe, test_size=0.2):
    """Shuffle and stratify the data by task, so that each task is represented equally in the train and test sets.

    Also ensures no subject is present only in the test set.
    If this is the case, half of the tasks of this subject are moved to the train set.
    """
    train_subjects, test_subjects = train_test_split(
        dataset_dataframe,
        test_size=test_size,
        stratify=dataset_dataframe["task"],
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


def training_loop(
    epochs,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    config,
):
    """Training loop."""
    history = {
        "epoch": 0,
        "loss_total": [],
        "loss_si": [],
        "loss_td": [],
        "acc_si": [],
        "acc_td": [],
        "val-loss_total": [],
        "val-loss_si": [],
        "val-loss_td": [],
        "val-acc_si": [],
        "val-acc_td": [],
    }
    print(f"Using {device}")
    if WANDB_AVAILABLE:
        wb.init(project="DLB-Project", config=config)
        wb.watch(model)

    model.to(device)

    for epoch in range(1, epochs + 1):
        start_epoch = time.time()
        loss_si, train_loss_td, total_loss, acc_si, acc_td = 0, 0, 0, 0, 0

        # Training
        model.train()
        for _i, batch in enumerate(
            train_loader
        ):  # NOTE: you can add tqdm(enumerate(train_loader)) to get a progress bar
            optimizer.zero_grad()

            p_matrix = batch[0].to(device)
            label_target_ids = batch[1].to(device)
            task_target_ids = batch[2].to(device)

            logits_si, logits_td, attention_weights = model(p_matrix)

            loss_si_c = criterion(
                logits_si, label_target_ids.long()
            )  # TODO(eddy) : maybe check the types ? Had to use .long() to avoid error
            loss_td_c = criterion(logits_td, task_target_ids.long())
            total_loss_c = (
                loss_si_c * config["lambda_si"]
                + loss_td_c * config["lambda_td"]
            )

            loss_si += loss_si_c.item()
            train_loss_td += loss_td_c.item()
            total_loss += total_loss_c.item()
            if WANDB_AVAILABLE:
                wb.log(
                    {
                        "Train/loss_si": loss_si_c.item(),
                        "Train/loss_td": loss_td_c.item(),
                        "Train/total_loss": total_loss_c.item(),
                    }
                )

            # logger.debug(f"{label_target_ids.shape}")
            # logger.debug(f"{logits_si.shape}")
            # logger.debug(f"{logits_td.shape}")

            pred_si = F.softmax(logits_si, dim=1).detach().cpu().numpy()
            pred_td = F.softmax(logits_td, dim=1).detach().cpu().numpy()

            # logger.debug(f"Pred SI for acc : {pred_si.shape}")
            # logger.debug(f"Pred TD for acc : {pred_td.shape}")
            # logger.debug(f"Preds SI : {pred_si}")

            acc_si += accuracy_score(
                y_true=label_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_si, axis=1),
            )
            acc_td += accuracy_score(
                y_true=task_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_td, axis=1),
            )

            if WANDB_AVAILABLE:
                wb.log(
                    {
                        "Train/acc_si": acc_si,
                        "Train/acc_td": acc_td,
                    }
                )

            total_loss_c.backward()
            optimizer.step()

        train_loss_total = total_loss / len(train_loader)
        train_loss_si = loss_si / len(train_loader)
        train_loss_td = train_loss_td / len(train_loader)
        train_acc_si = (acc_si / len(train_loader)) * 100
        train_acc_td = (acc_td / len(train_loader)) * 100

        logger.debug(f"SI loss : {train_loss_si}")
        logger.debug(f"TD loss : {train_loss_td}")

        if WANDB_AVAILABLE:
            wb.log(
                {
                    "Train/Epoch-loss_si": train_loss_si,
                    "Train/Epoch-loss_td": train_loss_td,
                    "Train/Epoch-total_loss": train_loss_total,
                }
            )

        if WANDB_AVAILABLE:
            wb.log(
                {
                    "Train/Epoch-acc_si": train_acc_si,
                    "Train/Epoch-acc_td": train_acc_td,
                }
            )

        # Validation
        (
            val_loss_total,
            val_loss_si,
            val_loss_td,
            val_acc_si,
            val_acc_td,
        ) = evaluate(model, valid_loader, criterion, device, config)

        if WANDB_AVAILABLE:
            wb.log(
                {
                    "Val/Epoch-loss_si": val_loss_si,
                    "Val/Epoch-loss_td": val_loss_td,
                    "Val/Epoch-total_loss": val_loss_total,
                    "Val/Epoch-acc_si": val_acc_si,
                    "Val/Epoch-acc_td": val_acc_td,
                }
            )

        # Logging
        history["epoch"] += 1
        history["loss_total"].append(train_loss_total)
        history["loss_si"].append(train_loss_si)
        history["loss_td"].append(train_loss_td)
        history["acc_si"].append(train_acc_si)
        history["val-loss_total"].append(val_loss_total)
        history["val-loss_si"].append(val_loss_si)
        history["val-loss_td"].append(val_loss_td)
        history["val-acc_si"].append(val_acc_si)
        history["val-acc_td"].append(val_acc_td)
        print(
            f"Epoch: {epoch}/{epochs} - loss_total: {train_loss_total:.4f}"
            + f" - acc: SI {train_acc_si:.2f}% / TD {train_acc_td:.2f}%"
            + f" - val-loss_total: {val_loss_total:.4f}"
            + f" - val-acc: SI {val_acc_si:.2f}% / TD {val_acc_td:.2f}%"
            + f" - ({time.time()-start_epoch:.2f}s/epoch)"
        )
    print("Finished Training.")
    return history


def evaluate(model, loader, criterion, device, config):
    """Evaluate the model on the dataloader."""
    loss_si, loss_td, total_loss, acc_si, acc_td = 0, 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            p_matrix = batch[0].to(device)
            label_target_ids = batch[1].to(device)
            task_target_ids = batch[2].to(device)

            logits_si, logits_td, attention_weights = model(p_matrix)

            loss_si_c = criterion(logits_si, label_target_ids.long())
            loss_td_c = criterion(logits_td, task_target_ids.long())
            total_loss_c = (
                loss_si_c * config["lambda_si"]
                + loss_td_c * config["lambda_td"]
            )

            loss_si += loss_si_c.item()
            loss_td += loss_td_c.item()
            total_loss += total_loss_c.item()

            pred_si = F.softmax(logits_si, dim=1).detach().cpu().numpy()
            pred_td = F.softmax(logits_td, dim=1).detach().cpu().numpy()

            acc_si += accuracy_score(
                y_true=label_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_si, axis=1),
            )
            acc_td += accuracy_score(
                y_true=task_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_td, axis=1),
            )

        val_loss_total = total_loss / len(loader)
        val_loss_si = loss_si / len(loader)
        val_loss_td = loss_td / len(loader)
        val_acc_si = (acc_si / len(loader)) * 100
        val_acc_td = (acc_td / len(loader)) * 100

    return val_loss_total, val_loss_si, val_loss_td, val_acc_si, val_acc_td
