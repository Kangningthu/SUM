import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import sigmoid_focal_loss


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou



def uncertainty_aware_dice_loss(y_pred, y_true, smooth=1, uncertain_map=None, uncertain_map_threshold=0.5):
    # Ensure the input tensors have shape (batch_size, channels, height, width)
    # y_true and y_pred should have the same shape
    y_pred = F.sigmoid(y_pred)
    # y_pred = torch.clamp(y_pred, min=0, max=1)

    # if uncertain_map is not None:
    #     # adjust the y_true based on the uncertain_map
    #     y_true = y_true * (1 - uncertain_map)
    #     y_pred = y_pred * (1 - uncertain_map)

    if uncertain_map is not None:
        # adjust the y_true and y_pred based on the uncertain_map
        y_true = y_true * ((1 - uncertain_map) > uncertain_map_threshold)
        y_pred = y_pred * ((1 - uncertain_map) > uncertain_map_threshold)


    # Compute the intersection and the sum of cardinalities per sample
    intersection = (y_true * y_pred).sum(dim=(1, 2, 3))
    cardinalities = y_true.sum(dim=(1, 2, 3)) + y_pred.sum(dim=(1, 2, 3))

    # Compute the DICE score per sample
    dice_scores = (2. * intersection + smooth) / (cardinalities + smooth)

    # print(dice_scores)

    # Return the DICE loss per sample
    return 1 - dice_scores



def uncertainty_aware_focal_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor, uncertain_map: torch.Tensor, alpha: float = 0.25, gamma: float = 2):
    """
    Compute the uncertainty-aware focal loss.

    This function calculates the focal loss and adjusts it based on an uncertainty map. 
    The focal loss is reduced in regions where the uncertainty is high, which can help 
    the model focus on more certain regions during training.

    Args:
        pred_mask (torch.Tensor): The predicted mask tensor of shape (N, H, W), 
                                  where N is the batch size, H is the height, W is the width, 
                                  and C is the number of channels.
        gt_mask (torch.Tensor): The ground truth mask tensor of shape (N, H, W.
        uncertain_map (torch.Tensor): The uncertainty map tensor of shape (N, H, W), 
                                      where higher values indicate higher uncertainty.
        alpha (float, optional): The alpha parameter for the focal loss. Default is 0.25.
        gamma (float, optional): The gamma parameter for the focal loss. Default is 2.

    Returns:
        torch.Tensor: The computed uncertainty-aware focal loss, averaged over the batch, 
                      height, and width dimensions.
    """
    # Calculate the focal loss without reduction
    loss_focal_tempt = sigmoid_focal_loss(
        pred_mask,
        gt_mask,
        gamma=gamma,
        alpha=alpha,
        reduction='none'
    )

    # Multiply the focal loss by (1 - uncertain_map) to reduce the loss in uncertain regions
    loss_focal_tempt *= (1 - uncertain_map)

    # Average the loss over the batch, height, and width dimensions
    loss_focal_tempt = loss_focal_tempt.mean(dim=(1, 2, 3))

    return loss_focal_tempt