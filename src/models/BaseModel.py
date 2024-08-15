from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

# loss functions and metrics
import torchmetrics
from segmentation_models_pytorch.losses import (DiceLoss, JaccardLoss,
                                                LovaszLoss)
from torchvision.ops import sigmoid_focal_loss
from common_utils.metrics import (
    label_sparsity,
    prediction_sparsity
)


class BaseModel(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        loss_function (str, optional): Loss function to use. Defaults to "".
        pos_class_weight (int, optional): Weight for positive class. Defaults to 236.
    """
    def __init__(
        self,
        loss_function: str = "",
        pos_class_weight: int = 236,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.loss_function = loss_function
        self.pos_class_weight = pos_class_weight

        # DEFINE METRICS
        self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        self.test_f1 = self.train_f1.clone()

        self.train_avg_precision = torchmetrics.AveragePrecision("binary")
        self.train_precision = torchmetrics.Precision("binary")
        self.train_recall = torchmetrics.Recall("binary")
        self.train_accuracy = torchmetrics.Accuracy("binary")
        self.train_iou = torchmetrics.JaccardIndex("binary")

        self.val_avg_precision = torchmetrics.AveragePrecision("binary")
        self.val_precision = torchmetrics.Precision("binary")
        self.val_recall = torchmetrics.Recall("binary")
        self.val_accuracy = torchmetrics.Accuracy("binary")
        self.val_iou = torchmetrics.JaccardIndex("binary")

        self.test_avg_precision = torchmetrics.AveragePrecision("binary")
        self.test_precision = torchmetrics.Precision("binary")
        self.test_recall = torchmetrics.Recall("binary")
        self.test_accuracy = torchmetrics.Accuracy("binary")
        self.test_iou = torchmetrics.JaccardIndex("binary")
        
        self.train_conf_mat = torchmetrics.ConfusionMatrix("binary")
        self.val_conf_mat = torchmetrics.ConfusionMatrix("binary")
        self.test_conf_mat = torchmetrics.ConfusionMatrix("binary")

        # Plot PR curve at the end of training. Use fixed number of threshold to avoid the plot becoming 800MB+. 
        self.train_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)
        self.val_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)

        self.metrics = {'train_f1':self.train_f1, 'val_f1':self.val_f1, 'test_f1':self.test_f1,
                        'train_avgprecision':self.train_avg_precision, 'val_avgprecision':self.val_avg_precision, 
                        'test_avgprecision':self.test_avg_precision, 'train_precision':self.train_precision, 
                        'val_precision':self.val_precision, 'test_precision':self.test_precision,
                        'train_recall':self.train_recall, 'val_recall':self.val_recall, 'test_recall':self.test_recall,
                        'train_iou':self.train_iou, 'val_iou':self.val_iou, 'test_iou':self.test_iou,
                        'train_accuracy':self.train_accuracy, 'val_accuracy':self.val_accuracy, 
                        'test_accuracy':self.test_accuracy
                        }

        self.criterion = self.get_loss()

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def get_loss(self):
        if self.loss_function  == "BCE":
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor(
                    [self.hparams.pos_class_weight], device=self.device
                )
            )
        elif self.loss_function == "Focal":
            return sigmoid_focal_loss
        elif self.loss_function == "Lovasz":
            return LovaszLoss(mode="binary")
        elif self.loss_function == "Jaccard":
            return JaccardLoss(mode="binary")
        elif self.loss_function == "Dice":
            return DiceLoss(mode="binary")

    def compute_loss(self, logits, y):
        if self.hparams.loss_function == "Focal":
            return self.loss(
                logits.squeeze(),
                y.float().squeeze(),
                alpha=1 - self.hparams.pos_class_weight,
                gamma=2,
                reduction="mean",
            )
        else:
            return self.loss(logits.squeeze(), y.float().squeeze())

    def training_step(self, batch: Any, batch_idx: int):
        self.net.train()
        
        x, y, lead_times, variables, out_variables = batch

        logits = self.net.forward(
            x, 
            y, 
            lead_times, 
            variables, 
            out_variables
            )
        
        loss = self.criterion(logits.squeeze().float(), y.squeeze().float())
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss_dict = {}
        loss_dict["loss"] = loss
        for var in self.metrics:
            if var.split('_')[0]=='train':
                self.metrics[var](logits.squeeze(), y.squeeze().long())
                self.log(
                    "train/" + var.split('_')[1],
                    self.metrics[var],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                loss_dict[var.split('_')[1]] = self.metrics[var]
        
        self.log(
            "train/label_sparsity",
            label_sparsity(y)['label_sparsity'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/prediction_sparsity",
            prediction_sparsity(logits)['prediction_sparsity'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.net.eval()
        
        x, y, lead_times, variables, out_variables = batch
        
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        logits = self.net.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            log_postfix=log_postfix
        )

        loss = self.criterion(logits.squeeze().float(), y.squeeze().float())
        self.log(
            "val/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss_dict = {}
        loss_dict["loss"] = loss
        for var in self.metrics:
            if var.split('_')[0]=='val':
                self.metrics[var](logits.squeeze(), y.squeeze().long())
                self.log(
                    "val/" + var.split('_')[1],
                    self.metrics[var],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                loss_dict[var.split('_')[1]] = self.metrics[var]

        self.log(
            "val/label_sparsity",
            label_sparsity(y)['label_sparsity'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val/prediction_sparsity",
            prediction_sparsity(logits)['prediction_sparsity'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):  
        self.net.eval()

        x, y, lead_times, variables, out_variables = batch
        
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        logits = self.net.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            log_postfix=log_postfix,
        )

        loss = self.criterion(logits.squeeze().float(), y.squeeze().float())
        self.log(
            "test/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss_dict = {}
        loss_dict["loss"] = loss.item()
        for var in self.metrics:
            if var.split('_')[0]=='test':
                self.metrics[var](logits.squeeze(), y.squeeze().long())
                self.log(
                    "test/" + var.split('_')[1],
                    self.metrics[var],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                loss_dict[var.split('_')[1]] = self.metrics[var]

        self.log(
            "test/label_sparsity",
            label_sparsity(y)['label_sparsity'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test/prediction_sparsity",
            prediction_sparsity(logits)['prediction_sparsity'],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss_dict