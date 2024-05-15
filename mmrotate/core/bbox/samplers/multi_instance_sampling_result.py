# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.samplers.sampling_result import SamplingResult


class MultiInstanceSamplingResult(SamplingResult):
    """Bbox sampling result. Further encapsulation of SamplingResult. Three
    attributes neg_assigned_gt_inds, neg_gt_labels, and neg_gt_bboxes have been
    added for SamplingResult.

    Args:
        pos_inds (Tensor): Indices of positive samples.
        neg_inds (Tensor): Indices of negative samples.
        priors (Tensor): The priors can be anchors or points,
            or the bboxes predicted by the previous stage.
        gt_and_ignore_bboxes (Tensor): Ground truth and ignore bboxes.
        assign_result (:obj:`AssignResult`): Assigning results.
        gt_flags (Tensor): The Ground truth flags.
        avg_factor_with_neg (bool):  If True, ``avg_factor`` equal to
            the number of total priors; Otherwise, it is the number of
            positive priors. Defaults to True.
    """

    def __init__(self,
                 pos_inds: Tensor,
                 neg_inds: Tensor,
                 priors: Tensor,
                 gt_and_ignore_bboxes: Tensor,
                 assign_result: AssignResult,
                 gt_flags: Tensor,
                 avg_factor_with_neg: bool = True) -> None:
        self.neg_assigned_gt_inds = assign_result.gt_inds[neg_inds]
        self.neg_gt_labels = assign_result.labels[neg_inds]

        if gt_and_ignore_bboxes.numel() == 0:
            self.neg_gt_bboxes = torch.empty_like(gt_and_ignore_bboxes).view(
                -1, 5)
        else:
            if len(gt_and_ignore_bboxes.shape) < 2:
                gt_and_ignore_bboxes = gt_and_ignore_bboxes.view(-1, 5)
            self.neg_gt_bboxes = gt_and_ignore_bboxes[
                self.neg_assigned_gt_inds.long(), :]

        # To resist the minus 1 operation in `SamplingResult.init()`.
        assign_result.gt_inds += 1
        super().__init__(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            bboxes=priors,
            gt_bboxes=gt_and_ignore_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags)
        
    @property
    def priors(self):
        """torch.Tensor: concatenated positive and negative priors"""
        return torch.cat([self.pos_bboxes, self.neg_bboxes], dim=0)
