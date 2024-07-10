from typing import Any
import wandb
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms
import torchmetrics
from segmentation_models_pytorch.losses import (DiceLoss, JaccardLoss,
                                                LovaszLoss)
from torchvision.ops import sigmoid_focal_loss

from models.climax.arch import ClimaX
from models.climax.pos_embed import interpolate_pos_embed
from models.climax.lr_scheduler import LinearWarmupCosineAnnealingLR
from common_utils.metrics import (
    binary_cross_entropy,
    confusion_matrix,
    label_sparsity,
    predicition_sparsity,
    auc,
    accuracy,
    iou,
    recall,
    avg_precision,
    precision,
    f1
)


class ClimaXModule(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        net (ClimaX): ClimaX model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """
    def __init__(
        self,
        net: ClimaX,
        pretrained_res: str = "",
        experiment: str = "",
        loss_function: str = "",
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        pos_class_weight: int = 236 
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.experiment = experiment
        self.pretrained_res = pretrained_res
        self.loss_function = loss_function
        self.pos_class_weight = pos_class_weight
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

        # DEFINE METRICS
        self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        self.test_f1 = self.train_f1.clone()

        self.train_avg_precision = torchmetrics.AveragePrecision("binary")
        self.train_precision = torchmetrics.Precision("binary")
        self.train_recall = torchmetrics.Recall("binary")
        self.train_iou = torchmetrics.JaccardIndex("binary")

        self.val_avg_precision = torchmetrics.AveragePrecision("binary")
        self.val_precision = torchmetrics.Precision("binary")
        self.val_recall = torchmetrics.Recall("binary")
        self.val_iou = torchmetrics.JaccardIndex("binary")

        self.test_avg_precision = torchmetrics.AveragePrecision("binary")
        self.test_precision = torchmetrics.Precision("binary")
        self.test_recall = torchmetrics.Recall("binary")
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
                        'train_iou':self.train_iou, 'val_iou':self.val_iou, 'test_iou':self.test_iou
                        }

        self.criterion = self.get_loss()

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. "/
                    "Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

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
        x, y, lead_times, variables, out_variables = batch

        _, logits = self.net.forward(
            x, 
            y, 
            lead_times, 
            variables, 
            out_variables
            )
        
        loss = self.criterion(logits.squeeze(), y.squeeze())
        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for var in self.metrics:
            if var.split('_')[0]=='train':
                self.metrics[var](logits.squeeze(), y.squeeze())
                self.log(
                    "train/" + var.split('_')[1],
                    self.metrics[var],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch
        
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        _, logits = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            log_postfix=log_postfix
        )

        loss = self.criterion(logits.squeeze(), y.squeeze())
        self.log(
            "val/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for var in self.metrics:
            if var.split('_')[0]=='val':
                self.metrics[var](logits.squeeze(), y.squeeze())
                self.log(
                    "val/" + var.split('_')[1],
                    self.metrics[var],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
        
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        
        x, y, lead_times, variables, out_variables = batch
        
        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts, logits = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            log_postfix=log_postfix,
        )

        loss = self.criterion(logits.squeeze(), y.squeeze())
        self.log(
            "test/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for var in self.metrics:
            if var.split('_')[0]=='test':
                self.metrics[var](logits.squeeze(), y.squeeze())
                self.log(
                    "test/" + var.split('_')[1],
                    self.metrics[var],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}