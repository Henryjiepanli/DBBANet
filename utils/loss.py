import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable

import re


EPSILON = 1e-12


class MultiHotNLLLoss(nn.Module):
    """
    Relax Loss

    inputs: (B, C, <spatial dims>)
    targets: (B, C, <spatial dims>) multi-hot categorical mask
    """

    def __init__(self, weights=None, equal_category_counts=True, reduction='mean'):
        super(MultiHotNLLLoss, self).__init__()
        self.weights = weights
        self.equal_category_counts = equal_category_counts
        self.reduction = reduction

    def forward(self, inputs, target):
        if self.equal_category_counts:
            counting_weights = target.sum(dim=1).float()
        else:
            counting_weights = 1.
        mask_invalid = (target.sum(dim=1) == 0).float()

        if self.weights is None:
            weights = 1.
        elif isinstance(self.weights, torch.Tensor):
            weights = self.weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        elif self.weights == 'batch_weighted':
            _dims = [0, *list(range(2, len(target.shape)))]
            weights = target.sum(dim=_dims) / (target.sum() + EPSILON)
            weights = torch.flip(weights, dims=[0])
            for _dim in _dims:
                weights = weights.unsqueeze(_dim)
        else:
            raise ValueError('Unknown weights \"%s\".' % self.weights)

        loss = -1 * (target.float() * inputs * weights).sum(dim=1) / counting_weights * (1 - mask_invalid)
        if self.reduction == 'mean':
            loss = loss.sum() / (np.prod(inputs.shape[2:]) * inputs.shape[0] - mask_invalid.sum() + EPSILON)
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError('Unknown reduction \"%s\"' % self.reduction)
        return loss


class MultiHotCrossEntropyLoss(nn.Module):
    """
    Relax Loss

    inputs: (B, C, <spatial dims>)
    targets: (B, C, <spatial dims>) multi-hot categorical mask
    """

    def __init__(self, weights=None, equal_category_counts=True, reduction='mean'):
        super(MultiHotCrossEntropyLoss, self).__init__()
        self.weights = weights
        self.equal_category_counts = equal_category_counts
        self.reduction = reduction
        self.multi_hot_nll_loss = MultiHotNLLLoss(weights, equal_category_counts, reduction)
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        loss = self.multi_hot_nll_loss(self.m(input), target)
        return loss


class OhemCrossEntropyLoss(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000, reduction='mean', ce_weight=None):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.reduction = reduction
        self.criterion = torch.nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, input, target):
        b, c = input.shape[0:2]
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(input, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view([b, *input.shape[2:]])

        return self.criterion(input, target)


class BatchWeightedBCELoss(nn.Module):
    def __init__(self, num_classes, reduction='mean', ignore_index=255):
        super(BatchWeightedBCELoss, self).__init__()
        self.num_classes = num_classes  # (only support num_classes=2)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        hist = categorical_batch_hist(target, self.num_classes)
        weights = torch.ones_like(target)

        if (target == 0).sum() > 0:
            weights[target == 0] = hist[1]
        if (target == 1).sum() > 0:
            weights[target == 1] = hist[0]
        weights[target == self.ignore_index] = 0
        # for c in range(self.num_classes):
        #     weights[target == self.num_classes - c - 1] = hist[c]  # ^
        weights = weights.float() / hist.sum()
        loss = F.binary_cross_entropy_with_logits(input, target, weight=weights, reduction=self.reduction)
        return loss


class NonDirectionalCosSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NonDirectionalCosSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        loss = 1 - torch.sign((input * target).sum(dim=1)) * F.cosine_similarity(input, target)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError('Unknown reduction \"%s\"' % self.reduction)
        return loss


class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, ignore_index=255, reduction='mean', gamma=2.0, normalize=False):
        super(CrossEntropyFocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, input, target):
        loss = self.softmax_focalloss(input, target, ignore_index=self.ignore_index,
                                      gamma=self.gamma, normalize=self.normalize)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError('Unknown reduction \"%s\"' % self.reduction)
        return loss

    @staticmethod
    def softmax_focalloss(y_pred, y_true, ignore_index=255, gamma=2.0, normalize=False):
        """

        Args:
            y_pred: [N, #class, H, W]
            y_true: [N, H, W] from 0 to #class
            gamma: scalar

        Returns:

        """
        losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
        with torch.no_grad():
            p = y_pred.softmax(dim=1)
            modulating_factor = (1 - p).pow(gamma)
            valid_mask = ~ y_true.eq(ignore_index)
            masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
            modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(
                dim=1)
            scale = 1.
            if normalize:
                scale = losses.sum() / (losses * modulating_factor).sum()
        losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

        return losses


def categorical_batch_hist(mask, num_classes):
    """
    mask: scalar categorical mask
    """
    return torch.histc(mask, bins=num_classes, min=0, max=num_classes-1)
