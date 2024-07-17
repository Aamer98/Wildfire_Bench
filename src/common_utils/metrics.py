import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
import torchmetrics
from scipy import stats
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    average_precision_score, 
    recall_score, 
    f1_score, 
    jaccard_score, 
    confusion_matrix, 
    auc
)



def confusion_matrix(logits, y, vars):
    """Confusion Matrix
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = confusion_matrix(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["confusion_matrix"] = scores

    return score_dict


def prediction_sparsity(logits):
    """Prediction Sparsity
    Args:
        logits: [B, L, V*p*p]
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    preds = preds.tolist()

    sparsity = preds.count(0) / len(preds)

    score_dict = {}
    score_dict["prediction_sparsity"] = sparsity

    return score_dict


def label_sparsity(y):
    """Label Sparsity
    Args:
        y: [B, V, H, W]
    """
    y = y.int().squeeze().flatten().cpu()
    y = y.tolist()

    sparsity = y.count(0) / len(y)

    score_dict = {}
    score_dict["label_sparisty"] = sparsity

    return score_dict


def auc(logits, y, vars):
    """auc
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = auc(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["auc"] = scores.mean()

    return score_dict


def accuracy(logits, y, vars):
    """Accuracy
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = accuracy_score(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["accuracy"] = scores.mean()

    return score_dict


def iou(logits, y, vars):
    """IoU
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = jaccard_score(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["iou"] = scores.mean()

    return score_dict


def recall(logits, y, vars):
    """Recall
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = recall_score(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["recall"] = scores.mean()

    return score_dict


def avg_precision(logits, y, vars):
    """Average precision
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = average_precision_score(y, preds)
    score_dict = {}
    score_dict["avg_precision"] = scores.mean()

    return score_dict


def precision(logits, y, vars):
    """Precision
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = precision_score(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["precision"] = scores.mean()

    return score_dict


def f1(logits, y, vars):
    """F1 score
    Args:
        logits: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    preds = torch.sigmoid(logits)
    preds = torch.round(preds).int()
    preds = preds.squeeze().flatten().detach().cpu()
    y = y.int().squeeze().flatten().cpu()
    
    scores = f1_score(y_true=y, y_pred=preds)
    score_dict = {}
    score_dict["f1"] = scores.mean()

    return score_dict


def binary_cross_entropy(pred, y, vars):
    """Binary Cross Entropy
    Args:
        pred: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    bce = binary_cross_entropy_with_logits
    loss = bce(pred ,y.float())

    loss_dict = {}   
    loss_dict["loss"] = loss.mean()

    return loss_dict


def mse(pred, y, vars, lat=None, mask=None):
    """Mean squared error
    Args:
        pred: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """
    loss = (pred - y) ** 2

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()

    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error
    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted root mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    pred = transform(pred)
    y = transform(y)
    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )
    
    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """
    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )
    
    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """
    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            y_ = y[:, i]  # B, H, W
            error = (torch.mean(pred_, dim=0) - torch.mean(y_, dim=0)) ** 2  # H, W
            error = torch.mean(error * w_lat)
            loss_dict[f"w_nrmses_{var}"] = torch.sqrt(error) / y_normalization
    
    return loss_dict


def lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """
    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            pred_ = torch.mean(pred_ * w_lat, dim=(-2, -1))  # B
            y_ = y[:, i]  # B, H, W
            y_ = torch.mean(y_ * w_lat, dim=(-2, -1))  # B
            error = torch.mean((pred_ - y_) ** 2)
            loss_dict[f"w_nrmseg_{var}"] = torch.sqrt(error) / y_normalization

    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """
    nrmses = lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix)
    nrmseg = lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix)
    
    loss_dict = {}
    for var in vars:
        loss_dict[f"w_nrmses_{var}"] = nrmses[f"w_nrmses_{var}"]
        loss_dict[f"w_nrmseg_{var}"] = nrmseg[f"w_nrmseg_{var}"]
        loss_dict[f"w_nrmse_{var}"] = nrmses[f"w_nrmses_{var}"] + 5 * nrmseg[f"w_nrmseg_{var}"]
    
    return loss_dict


def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt


def pearson(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """
    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_, y_ = pred[:, step - 1, i].flatten(), y[:, step - 1, i].flatten()
                pred_, y_ = remove_nans(pred_, y_)
                loss_dict[f"pearsonr_{var}_day_{day}"] = stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    loss_dict["pearsonr"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_mean_bias(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """
    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_, y_ = pred[:, step - 1, i].flatten(), y[:, step - 1, i].flatten()
                pred_, y_ = remove_nans(pred_, y_)
                loss_dict[f"mean_bias_{var}_day_{day}"] = pred_.mean() - y_.mean()

                # pred_mean = torch.mean(w_lat * pred[:, step - 1, i])
                # y_mean = torch.mean(w_lat * y[:, step - 1, i])
                # loss_dict[f"mean_bias_{var}_day_{day}"] = y_mean - pred_mean

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict