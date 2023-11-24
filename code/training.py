"""Functions for training and evaluating models."""
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False  # TODO(cyril) : add wandb to training

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
        ~test_subjects["subject"].isin(train_subjects["subject"])
    ]
    if len(test_only_subjects) > 0:
        print(
            f"Found {len(test_only_subjects['subject'].unique())} subjects present only in the test set"
        )
        # if there are subjects present only in the test set, move half of their tasks to the train set
        for subject in test_only_subjects["subject"].unique():
            subject_tasks = test_subjects[
                test_subjects["subject"] == subject
            ].sample(frac=0.5)
            train_subjects = train_subjects.append(subject_tasks)
            test_subjects = test_subjects.drop(subject_tasks.index)
            print(
                f"Moved {len(subject_tasks)} tasks from subject {subject} to the train set"
            )


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

    for epoch in range(1, epochs + 1):
        start_epoch = time.time()
        loss_si, loss_td, total_loss = 0, 0, 0

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
            loss_td += loss_td_c.item()
            total_loss += total_loss_c.item()

            total_loss_c.backward()
            optimizer.step()

        train_loss_total = total_loss / len(train_loader)
        train_loss_si = loss_si / len(train_loader)
        loss_td = loss_td / len(train_loader)

        logger.debug(f"SI loss : {train_loss_si}")
        logger.debug(f"TD loss : {loss_td}")

        # logger.debug(f"{label_target_ids.shape}")
        # logger.debug(f"{logits_si.shape}")
        # logger.debug(f"{logits_td.shape}")
        pred_si = F.softmax(logits_si, dim=1).detach().cpu().numpy()
        pred_td = F.softmax(logits_td, dim=1).detach().cpu().numpy()

        # logger.debug(f"Pred SI for acc : {pred_si.shape}")
        # logger.debug(f"Pred TD for acc : {pred_td.shape}")
        # logger.debug(f"Preds SI : {pred_si}")

        train_acc_si = accuracy_score(
            y_true=label_target_ids.detach().cpu().squeeze().numpy(),
            y_pred=np.argmax(pred_si, axis=1),
        )
        train_acc_td = accuracy_score(
            y_true=task_target_ids.detach().cpu().squeeze().numpy(),
            y_pred=np.argmax(pred_td, axis=1),
        )

        # Validation
        (
            val_loss_total,
            val_loss_si,
            val_loss_td,
            val_acc_si,
            val_acc_td,
        ) = evaluate(model, valid_loader, criterion, device, config)

        # Logging
        history["epoch"] += 1
        history["loss_total"].append(train_loss_total)
        history["loss_si"].append(train_loss_si)
        history["loss_td"].append(loss_td)
        history["acc_si"].append(train_acc_si)
        history["val-loss_total"].append(val_loss_total)
        history["val-loss_si"].append(val_loss_si)
        history["val-loss_td"].append(val_loss_td)
        history["val-acc_si"].append(val_acc_si)
        history["val-acc_td"].append(val_acc_td)
        print(
            f"Epoch: {epoch}/{epochs} - loss_total: {train_loss_total:.4f}"
            + f"- Acc: SI {train_acc_si:.4f} / TD {train_acc_td:.4f}"
            + f"- val-loss_total: {val_loss_total:.4f} - val-acc: SI {val_acc_si:.4f} TD {val_acc_td:.4f}"
            + f"({time.time()-start_epoch:.2f}s/epoch)"
        )
    print("Finished Training.")
    return history


def evaluate(model, loader, criterion, device, config):
    """Evaluate the model on the dataloader."""
    loss_si, loss_td, total_loss = 0, 0, 0
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

        test_loss_total = total_loss / len(loader)
        test_loss_si = loss_si / len(loader)
        loss_td = loss_td / len(loader)

        pred_si = F.softmax(logits_si, dim=1).detach().cpu().numpy()
        pred_td = F.softmax(logits_td, dim=1).detach().cpu().numpy()

        acc_si = accuracy_score(
            y_true=label_target_ids.detach().cpu().squeeze().numpy(),
            y_pred=np.argmax(pred_si, axis=1),
        )
        acc_td = accuracy_score(
            y_true=task_target_ids.detach().cpu().squeeze().numpy(),
            y_pred=np.argmax(pred_td, axis=1),
        )
    return test_loss_total, test_loss_si, loss_td, acc_si, acc_td
