"""
Loss functions for InterActVLM-Discrete (IVD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiceLoss(nn.Module):
    """Dice loss for mask prediction."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: (B, 1, H, W) predicted mask (after sigmoid)
            target: (B, 1, H, W) ground truth mask
            
        Returns:
            Scalar loss
        """
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: (B, N) predicted logits
            target: (B, N) binary targets
            
        Returns:
            Scalar loss
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ChamferLoss(nn.Module):
    """Chamfer distance loss for point set prediction."""
    
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Chamfer distance.
        
        Args:
            pred: (B, K, 3) predicted points
            target: (B, K, 3) target points
            mask: (B, K) optional mask for valid points
            
        Returns:
            Scalar loss
        """
        # pred -> target distance
        dist_p2t = torch.cdist(pred, target)  # (B, K, K)
        min_p2t = dist_p2t.min(dim=2)[0]  # (B, K)
        
        if mask is not None:
            min_p2t = min_p2t * mask
            loss_p2t = min_p2t.sum() / (mask.sum() + 1e-8)
        else:
            loss_p2t = min_p2t.mean()
        
        if self.bidirectional:
            # target -> pred distance
            min_t2p = dist_p2t.min(dim=1)[0]  # (B, K)
            
            if mask is not None:
                min_t2p = min_t2p * mask
                loss_t2p = min_t2p.sum() / (mask.sum() + 1e-8)
            else:
                loss_t2p = min_t2p.mean()
            
            return (loss_p2t + loss_t2p) / 2
        else:
            return loss_p2t


class IVDLoss(nn.Module):
    """
    Combined loss function for InterActVLM-Discrete.
    
    Combines:
    1. L_human: Binary classification loss for 87 body points
    2. L_object: Index classification loss for object points
    3. L_aux_mask: Auxiliary mask prediction loss (for early training)
    """
    
    def __init__(
        self,
        lambda_human: float = 1.0,
        lambda_object: float = 1.0,
        lambda_aux_mask: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        object_ignore_index: int = -100,
        object_label_smoothing: float = 0.0,
        object_soft_label_tau: float = 0.02,
        lambda_object_aff_consistency: float = 0.2,
        lambda_human_aff_consistency: float = 0.2,
        aff_consistency_eps: float = 1e-6
    ):
        """
        Initialize IVD loss.
        
        Args:
            lambda_human: Weight for human contact loss
            lambda_object: Weight for object index loss
            lambda_aux_mask: Weight for auxiliary mask loss
            use_focal: Whether to use focal loss for human contacts
            use_chamfer: Deprecated, kept for backward compatibility
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
            object_ignore_index: Ignore index for invalid object targets
            object_label_smoothing: Label smoothing for object CE loss
            object_soft_label_tau: Temperature for soft-label loss (0=disabled)
            lambda_object_aff_consistency: Weight for affordance consistency loss
            lambda_human_aff_consistency: Weight for human affordance consistency loss
            aff_consistency_eps: Numerical stability for log
        """
        super().__init__()
        
        self.lambda_human = lambda_human
        self.lambda_object = lambda_object
        self.lambda_aux_mask = lambda_aux_mask
        self.lambda_object_aff_consistency = lambda_object_aff_consistency
        self.lambda_human_aff_consistency = lambda_human_aff_consistency
        self.aff_consistency_eps = aff_consistency_eps
        self.object_soft_label_tau = object_soft_label_tau
        
        # Human contact loss
        self.human_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        # Object index loss
        self.object_loss = nn.CrossEntropyLoss(
            ignore_index=object_ignore_index,
            label_smoothing=object_label_smoothing
        )
        
        # Auxiliary mask loss
        self.mask_bce = nn.BCEWithLogitsLoss()
        self.mask_dice = DiceLoss()
    
    def compute_human_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute human contact classification loss.
        
        Args:
            pred_logits: (B, 87) predicted logits
            target: (B, 87) binary ground truth
            
        Returns:
            Scalar loss
        """
        return self.human_loss(pred_logits, target)
    
    def compute_object_loss(
        self,
        pred_logits: torch.Tensor,
        target_indices: torch.Tensor,
        object_points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute object point index classification loss.
        
        Args:
            pred_logits: (B, K, N_o) predicted logits
            target_indices: (B, K) target point indices
            object_points: (B, N_o, 3) object point cloud for soft labels
            
        Returns:
            Scalar loss
        """
        B, K, N_o = pred_logits.shape

        if self.object_soft_label_tau > 0.0 and object_points is not None:
            gt_idx = target_indices.long().clamp(min=0)
            gt_points = torch.gather(
                object_points, 1, gt_idx.unsqueeze(-1).expand(-1, -1, 3)
            )  # (B, K, 3)
            d2 = (object_points.unsqueeze(1) - gt_points.unsqueeze(2)).pow(2).sum(dim=-1)  # (B, K, N_o)
            soft_targets = torch.softmax(-d2 / self.object_soft_label_tau, dim=-1)

            logp = torch.log_softmax(pred_logits, dim=-1)
            per = -(soft_targets * logp).sum(dim=-1)  # (B, K)
            mask = (target_indices != -100).float()
            denom = mask.sum().clamp(min=1.0)
            return (per * mask).sum() / denom

        logits_flat = pred_logits.reshape(B * K, N_o)
        target_flat = target_indices.reshape(B * K).long()
        return self.object_loss(logits_flat, target_flat)

    def compute_affordance_consistency_loss(
        self,
        index_logits: torch.Tensor,
        object_affordance_logits: torch.Tensor,
        human_labels: torch.Tensor,
        object_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encourage object index predictions to align with object affordance scores.

        Args:
            index_logits: (B, K, N_o) index logits
            object_affordance_logits: (B, N_o) affordance logits
            human_labels: (B, K) human contact labels (supervise positives only)
            object_indices: (B, K) target indices (-100 for ignore), optional

        Returns:
            Scalar loss
        """
        if object_affordance_logits.dim() != 2:
            raise ValueError("object_affordance_logits must be (B, N_o)")
        if index_logits.dim() != 3:
            raise ValueError("index_logits must be (B, K, N_o)")
        if human_labels is None:
            raise ValueError("human_labels is required for affordance consistency loss")
        
        p = torch.softmax(index_logits, dim=-1)  # (B, K, N_o)
        m = torch.sigmoid(object_affordance_logits).unsqueeze(1)  # (B, 1, N_o)
        mass = (p * m).sum(dim=-1)  # (B, K)
        mask = (human_labels > 0.5).float()
        if object_indices is not None:
            mask = mask * (object_indices != -100).float()

        denom = mask.sum().clamp(min=1.0)
        loss = -(torch.log(mass + self.aff_consistency_eps) * mask).sum() / denom
        return loss

    def compute_human_affordance_consistency_loss(
        self,
        human_affordance_logits: torch.Tensor,
        human_keypoint_indices: torch.Tensor,
        human_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage human keypoints with positive labels to lie in high-affordance region.

        Args:
            human_affordance_logits: (B, N_h) human affordance logits
            human_keypoint_indices: (B, K) indices into N_h vertices
            human_labels: (B, K) binary keypoint labels

        Returns:
            Scalar loss
        """
        if human_affordance_logits.dim() != 2:
            raise ValueError("human_affordance_logits must be (B, N_h)")
        if human_keypoint_indices.dim() != 2:
            raise ValueError("human_keypoint_indices must be (B, K)")
        if human_labels.dim() != 2:
            raise ValueError("human_labels must be (B, K)")

        probs = torch.sigmoid(human_affordance_logits)
        kp_probs = torch.gather(probs, 1, human_keypoint_indices.long())  # (B, K)
        mask = (human_labels > 0.5).float()
        denom = mask.sum().clamp(min=1.0)
        loss = -(torch.log(kp_probs + self.aff_consistency_eps) * mask).sum() / denom
        return loss
    
    def compute_mask_loss(
        self,
        pred_mask: torch.Tensor,
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary mask loss (Dice + BCE).
        
        Args:
            pred_mask: (B, 1, H, W) predicted mask logits
            target_mask: (B, 1, H, W) ground truth mask
            
        Returns:
            Scalar loss
        """
        if target_mask.device != pred_mask.device:
            target_mask = target_mask.to(pred_mask.device)
        if target_mask.dtype != pred_mask.dtype:
            target_mask = target_mask.to(dtype=pred_mask.dtype)

        bce_loss = self.mask_bce(pred_mask, target_mask)
        
        pred_sigmoid = torch.sigmoid(pred_mask)
        dice_loss = self.mask_dice(pred_sigmoid, target_mask)
        
        return bce_loss + dice_loss
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        compute_aux: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            predictions: Dictionary with:
                - 'human_logits': (B, 87) predicted contact logits
                - 'human_contact': (B, 87) predicted contact probabilities
                - 'object_index_logits': (B, K, N_o) predicted index logits
                - 'object_index_probs': (B, K, N_o) predicted index probabilities
                - 'human_mask': (B*J, 1, H, W) auxiliary human mask (optional)
                - 'object_mask': (B*J, 1, H, W) auxiliary object mask (optional)
                - 'object_affordance_logits': (B, N_o) object affordance logits (optional)
                - 'human_affordance_logits': (B, N_h) human affordance logits (optional)
            targets: Dictionary with:
                - 'human_labels': (B, 87) binary contact labels
                - 'object_indices': (B, K) target point indices
                - 'object_points': (B, N_o, 3) object point cloud (optional)
                - 'human_keypoint_indices': (B, 87) human vertex indices
                - 'human_mask_gt': (B*J, 1, H, W) ground truth human mask (optional)
                - 'object_mask_gt': (B*J, 1, H, W) ground truth object mask (optional)
            compute_aux: Whether to compute auxiliary mask loss
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Human contact loss
        if 'human_logits' in predictions and 'human_labels' in targets:
            losses['human_loss'] = self.compute_human_loss(
                predictions['human_logits'],
                targets['human_labels']
            )
        else:
            losses['human_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Object index loss
        if 'object_index_logits' in predictions and 'object_indices' in targets:
            losses['object_loss'] = self.compute_object_loss(
                predictions['object_index_logits'],
                targets['object_indices'],
                targets.get('object_points', None)
            )
        else:
            losses['object_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # Auxiliary mask losses
        if compute_aux:
            if 'human_mask' in predictions and 'human_mask_gt' in targets:
                losses['human_mask_loss'] = self.compute_mask_loss(
                    predictions['human_mask'],
                    targets['human_mask_gt']
                )
            else:
                losses['human_mask_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            
            if 'object_mask' in predictions and 'object_mask_gt' in targets:
                losses['object_mask_loss'] = self.compute_mask_loss(
                    predictions['object_mask'],
                    targets['object_mask_gt']
                )
            else:
                losses['object_mask_loss'] = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            
            aux_loss = losses['human_mask_loss'] + losses['object_mask_loss']
        else:
            aux_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
            losses['human_mask_loss'] = aux_loss
            losses['object_mask_loss'] = aux_loss

        # Affordance consistency loss (optional)
        if self.lambda_object_aff_consistency > 0.0:
            if 'object_index_logits' in predictions and 'object_affordance_logits' in predictions and 'human_labels' in targets:
                losses['object_aff_consistency_loss'] = self.compute_affordance_consistency_loss(
                    predictions['object_index_logits'],
                    predictions['object_affordance_logits'],
                    targets['human_labels'],
                    targets.get('object_indices', None)
                )
            else:
                losses['object_aff_consistency_loss'] = torch.tensor(
                    0.0, device=next(iter(predictions.values())).device
                )
        else:
            losses['object_aff_consistency_loss'] = torch.tensor(
                0.0, device=next(iter(predictions.values())).device
            )

        if self.lambda_human_aff_consistency > 0.0:
            if 'human_affordance_logits' in predictions and 'human_keypoint_indices' in targets and 'human_labels' in targets:
                losses['human_aff_consistency_loss'] = self.compute_human_affordance_consistency_loss(
                    predictions['human_affordance_logits'],
                    targets['human_keypoint_indices'],
                    targets['human_labels']
                )
            else:
                losses['human_aff_consistency_loss'] = torch.tensor(
                    0.0, device=next(iter(predictions.values())).device
                )
        else:
            losses['human_aff_consistency_loss'] = torch.tensor(
                0.0, device=next(iter(predictions.values())).device
            )
        
        # Total loss
        losses['total_loss'] = (
            self.lambda_human * losses['human_loss'] +
            self.lambda_object * losses['object_loss'] +
            self.lambda_aux_mask * aux_loss +
            self.lambda_object_aff_consistency * losses['object_aff_consistency_loss'] +
            self.lambda_human_aff_consistency * losses['human_aff_consistency_loss']
        )
        
        return losses


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss for imbalanced binary classification."""
    
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, N) predicted logits
            target: (B, N) binary targets
        """
        pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        return F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )


def compute_contact_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute contact prediction accuracy metrics.
    
    Args:
        pred: (B, 87) predicted probabilities
        target: (B, 87) binary targets
        threshold: Classification threshold
        
    Returns:
        Dictionary with accuracy metrics
    """
    pred_binary = (pred > threshold).float()
    
    # Overall accuracy
    accuracy = (pred_binary == target).float().mean().item()
    
    # Per-point accuracy
    per_point_acc = (pred_binary == target).float().mean(dim=0)
    
    # Precision, Recall for positive class
    tp = ((pred_binary == 1) & (target == 1)).float().sum()
    fp = ((pred_binary == 1) & (target == 0)).float().sum()
    fn = ((pred_binary == 0) & (target == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'per_point_acc_mean': per_point_acc.mean().item()
    }
