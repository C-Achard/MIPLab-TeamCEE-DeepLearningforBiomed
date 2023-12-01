"""Functions for training and evaluating models."""
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from captum.attr import DeepLift
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

try:
    import wandb as wb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


def training_loop(
    epochs,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    config,
    scheduler=None,
    test_loader=None,
    save_model=False,
    save_attention_weights=False,
    run_name=None,
    use_deeplift=False,
    use_early_stopping=False,
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
        "LR": [],
    }
    print(f"Using {device}")
    if WANDB_AVAILABLE:
        if run_name is not None:
            wb.init(project="DLB-Project", config=config, name=run_name)
        else:
            wb.init(project="DLB-Project", config=config)
        wb.watch(model)

    if scheduler is not None:
        print("Using scheduler")

    model.to(device)

    for epoch in range(1, epochs + 1):
        start_epoch = time.time()
        loss_si, train_loss_td, total_loss, acc_si, acc_td = 0, 0, 0, 0, 0
        final_epoch_attention_weights = []
        # Training
        model.train()
        for _i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            p_matrix = batch[0].to(device)
            label_target_ids = batch[1].to(device)
            task_target_ids = batch[2].to(device)

            logits_si, logits_td, attention_weights = model(p_matrix)
            if epoch == epochs:
                final_epoch_attention_weights.append(attention_weights)

            # create heatmap from attention weights and log to wandb
            if WANDB_AVAILABLE and epoch % 20 == 0 and _i == 0:
                heatmap_att = sns.heatmap(
                    np.mean(
                        attention_weights.squeeze().detach().cpu().numpy(),
                        axis=0,
                    ),
                    annot=False,
                    cmap="turbo",
                    xticklabels=False,
                    yticklabels=False,
                    cbar=False,
                )
                wb.log({"Att/attention_weights": wb.Image(heatmap_att)})
                # heatmap_att_output = sns.heatmap(
                #     np.mean(
                #         torch.matmul(attention_weights, p_matrix).squeeze().detach().cpu().numpy(),
                #         axis=0
                #     ),
                #     annot=False,
                #     cmap="turbo",
                #     xticklabels=False,
                #     yticklabels=False,
                #     cbar=False
                # )
                # wb.log({"Att/attention_output": wb.Image(heatmap_att_output)})

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
        if scheduler is not None:
            scheduler.step()
            if WANDB_AVAILABLE:
                wb.log({"LR/LR": optimizer.param_groups[0]["lr"]})

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
            val_f1_si,
            val_f1_td,
        ) = evaluate(model, valid_loader, criterion, device, config)

        # Early stopping to avoid overfitting
        if use_early_stopping:
            if val_loss_total < config["best_loss"]:
                config["best_loss"] = val_loss_total
                patience = config["patience"]
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping")
                    break

        if WANDB_AVAILABLE:
            wb.log(
                {
                    "Val/Epoch-loss_si": val_loss_si,
                    "Val/Epoch-loss_td": val_loss_td,
                    "Val/Epoch-total_loss": val_loss_total,
                    "Val/Epoch-acc_si": val_acc_si,
                    "Val/Epoch-acc_td": val_acc_td,
                    "Val/Epoch-f1_si": val_f1_si,
                    "Val/Epoch-f1_td": val_f1_td,
                }
            )

        if (
            save_model
            and val_f1_si > history["val-acc_si"][-1]
            and val_f1_td > history["val-acc_td"][-1]
        ):
            torch.save(model.state_dict(), "best_val_model.pth")
            if WANDB_AVAILABLE:
                wb.save("best_val_model.pth")

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
        history["LR"].append(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch: {epoch}/{epochs} - loss_total: {train_loss_total:.4f}"
            + f" - acc: SI {train_acc_si:.2f}% / TD {train_acc_td:.2f}%\n"
            + f" - val-loss_total: {val_loss_total:.4f}"
            + f" - val-acc: SI {val_acc_si:.2f}% / TD {val_acc_td:.2f}%"
            + f" - val-f1: SI {val_f1_si:.4f} / TD {val_f1_td:.4f}"
            + f" - ({time.time()-start_epoch:.2f}s/epoch)"
        )

    if test_loader is not None:
        (
            (
                test_loss_total,
                test_loss_si,
                test_loss_td,
                test_acc_si,
                test_acc_td,
                test_f1_si,
                test_f1_td,
            ),
            test_labels,
            test_preds,
        ) = evaluate(
            model,
            test_loader,
            criterion,
            device,
            config,
            return_preds_td=True,
        )
        print("_" * 30)
        print(
            f"Final test loss: {test_loss_total:.4f} - acc: SI {test_acc_si:.2f}% / TD {test_acc_td:.2f}% - f1: SI {test_f1_si:.4f} / TD {test_f1_td:.4f}"
        )
        print("_" * 30)
        if WANDB_AVAILABLE:
            wb.log(
                {
                    "Test/loss_si": test_loss_si,
                    "Test/loss_td": test_loss_td,
                    "Test/total_loss": test_loss_total,
                    "Test/acc_si": test_acc_si,
                    "Test/acc_td": test_acc_td,
                    "Test/f1_si": test_f1_si,
                    "Test/f1_td": test_f1_td,
                }
            )
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
        wb.log(
            {
                "Test/confusion_matrix": wb.plot.confusion_matrix(
                    probs=None,
                    y_true=test_labels,
                    preds=test_preds,
                    class_names=task_labels,
                )
            }
        )
    if save_model:
        torch.save(model.state_dict(), "model.pth")
        if WANDB_AVAILABLE:
            wb.save("model.pth")

    if save_attention_weights:
        if not len(final_epoch_attention_weights):
            print("No attention weights saved, as there are none to save.")
        final_epoch_attention_weights_save = (
            torch.cat(final_epoch_attention_weights).detach().cpu().numpy()
        )
        np.save("attention_weights.npy", final_epoch_attention_weights_save)
        
    if use_deeplift:  # MUST BE KEPT AS LAST STEP
        print("Running DeepLIFT")
        model.eval()
        model._deeplift_mode = "si"
        dl = DeepLift(model)
        attributions_si = dl.attribute(
            inputs=p_matrix,
            baselines=torch.zeros_like(p_matrix),
            target=0,
        )
        print("SI attributions shape : ", attributions_si.shape)
        model._deeplift_mode = "td"
        dl = DeepLift(model)
        attributions_td = dl.attribute(
            inputs=p_matrix,
            baselines=torch.zeros_like(p_matrix),
            target=tuple(range(model.output_size_tasks)),
        )
        print("TD attributions shape : ", attributions_td.shape)
        attributions = (
            attributions_si.detach().cpu().numpy(), 
            attributions_td.detach().cpu().numpy()
            )
        with Path("attributions.pkl").open("wb") as f:
            pickle.dump(attributions, f)

    print("Finished Training.")
    return history


def evaluate(
    model,
    loader,
    criterion,
    device,
    config,
    return_preds_td=False,
):
    """Evaluate the model on the dataloader."""
    loss_si, loss_td, total_loss, acc_si, acc_td = 0, 0, 0, 0, 0
    f1_si, f1_td = 0, 0
    model.eval()

    if return_preds_td:
        labels_td = []
        preds_td = []

    with torch.no_grad():
        for batch in loader:
            p_matrix = batch[0].to(device)
            label_target_ids = batch[1].to(device)
            task_target_ids = batch[2].to(device)

            logits_si, logits_td, _ = model(p_matrix)

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
            f1_si += f1_score(
                y_true=label_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_si, axis=1),
                average="macro",
            )
            acc_td += accuracy_score(
                y_true=task_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_td, axis=1),
            )
            f1_td += f1_score(
                y_true=task_target_ids.detach().cpu().squeeze().numpy(),
                y_pred=np.argmax(pred_td, axis=1),
                average="macro",
            )

            if return_preds_td:
                labels_td.append(
                    task_target_ids.detach().cpu().squeeze().numpy()
                )
                preds_td.append(np.argmax(pred_td, axis=1))

        val_loss_total = total_loss / len(loader)
        val_loss_si = loss_si / len(loader)
        val_loss_td = loss_td / len(loader)
        val_acc_si = (acc_si / len(loader)) * 100
        val_acc_td = (acc_td / len(loader)) * 100
        val_f1_si = f1_si / len(loader)
        val_f1_td = f1_td / len(loader)

    if return_preds_td:
        return (
            (
                val_loss_total,
                val_loss_si,
                val_loss_td,
                val_acc_si,
                val_acc_td,
                val_f1_si,
                val_f1_td,
            ),
            np.concatenate(labels_td),
            np.concatenate(preds_td),
        )

    return (
        val_loss_total,
        val_loss_si,
        val_loss_td,
        val_acc_si,
        val_acc_td,
        val_f1_si,
        val_f1_td,
    )
