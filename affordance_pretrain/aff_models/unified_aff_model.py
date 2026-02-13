"""
Unified Affordance Model for Pre-training
Supports joint training with partial annotations (human-only or object-only)
Module names are kept consistent with IVDModel for weight loading
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.vlm_module import VLMModule, LightweightVLM
from models.pointnet2_encoder import PointNetv2Encoder
from models.pointnet2_decoder import PointNet2FeatureDecoder, TemplatePointProvider, AffordanceDecoder


class UnifiedAffordanceModel(nn.Module):
    """
    Unified Affordance Model for pre-training human and object affordance.

    Key Design:
    - Module names are IDENTICAL to IVDModel for direct weight loading
    - Supports partial annotations via valid masks
    - Can train with human-only, object-only, or mixed data

    Modules (matching IVDModel):
        - vlm: VLM semantic extraction
        - object_pc_encoder: PointNet++ encoder for object
        - object_point_decoder: PointNet++ decoder for object
        - point_feat_proj: Feature projection layer
        - object_sem_film: FiLM layer for semantic fusion
        - human_point_provider: Template point provider for human
        - human_affordance: AffordanceDecoder for human
        - object_affordance: AffordanceDecoder for object
    """

    def __init__(
        self,
        d_tr: int = 256,
        num_human_points: int = 10475,
        use_lightweight_vlm: bool = True,
        vlm_model_name: str = "llava-hf/llava-1.5-7b-hf",
        lora_r: int = 16,
        lora_alpha: int = 32,
        freeze_vlm: bool = True,
        dropout: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize Unified Affordance Model.

        Args:
            d_tr: Transformer/feature dimension
            num_human_points: Number of human template points (10475 for SMPL-X)
            use_lightweight_vlm: Use lightweight CLIP-based VLM
            vlm_model_name: VLM model name (if not lightweight)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            freeze_vlm: Whether to freeze VLM backbone
            dropout: Dropout rate
            device: Device
        """
        super().__init__()

        self.d_tr = d_tr
        self.num_human_points = num_human_points
        self.device = device

        # ============ VLM Module (shared) ============
        # Module name: vlm
        if use_lightweight_vlm:
            self.vlm = LightweightVLM(
                model_name="openai/clip-vit-base-patch32",
                d_tr=d_tr,
                device=device
            )
        else:
            self.vlm = VLMModule(
                model_name=vlm_model_name,
                d_tr=d_tr,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                freeze_backbone=freeze_vlm,
                device=device
            )

        # ============ Object Branch ============
        # Module name: object_pc_encoder
        self.object_pc_encoder = PointNetv2Encoder(with_decoder=False)

        # Module name: object_point_decoder
        self.object_point_decoder = PointNet2FeatureDecoder()

        # Module name: point_feat_proj
        self.point_feat_proj = nn.Linear(256, d_tr)

        # Module name: object_sem_film (shared for both human and object)
        self.object_sem_film = nn.Linear(d_tr, d_tr * 2)

        # Module name: object_affordance
        self.object_affordance = AffordanceDecoder(
            d_model=d_tr,
            context_dim=d_tr,
            dropout=dropout
        )

        # ============ Human Branch ============
        # Module name: human_point_provider
        self.human_point_provider = TemplatePointProvider(
            num_points=num_human_points,
            d_model=d_tr,
            context_dim=d_tr,
            dropout=dropout
        )

        # Module name: human_affordance
        self.human_affordance = AffordanceDecoder(
            d_model=d_tr,
            context_dim=d_tr,
            dropout=dropout
        )

        if freeze_vlm:
            self.freeze_vlm()

    @staticmethod
    def _fuse_semantic(
        point_feat: torch.Tensor,
        sem_emb: torch.Tensor,
        film: nn.Module
    ) -> torch.Tensor:
        """Apply FiLM modulation to fuse semantic embedding with point features."""
        gamma_beta = film(sem_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return point_feat * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(
        self,
        rgb_image: torch.Tensor,
        object_points: Optional[torch.Tensor] = None,
        compute_human: bool = True,
        compute_object: bool = True,
        text_prompts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            rgb_image: (B, 3, 224, 224) RGB image
            object_points: (B, N, 3) object point cloud (optional)
            compute_human: Whether to compute human affordance
            compute_object: Whether to compute object affordance
            text_prompts: Optional text prompts for VLM

        Returns:
            Dictionary with:
                - 'human_affordance_logits': (B, 10475) human affordance logits
                - 'human_affordance': (B, 10475) human affordance probs
                - 'object_affordance_logits': (B, N) object affordance logits
                - 'object_affordance': (B, N) object affordance probs
        """
        outputs = {}

        # VLM semantic extraction (always run)
        E_human, E_object = self.vlm(rgb_image, text_prompts)

        # Human branch
        if compute_human:
            human_points = self.human_point_provider(E_human)
            human_feat = human_points['features']  # (B, 10475, d_tr)
            human_feat = self._fuse_semantic(human_feat, E_human, self.object_sem_film)

            human_aff = self.human_affordance(human_feat, E_human)
            outputs['human_affordance_logits'] = human_aff['logits']
            outputs['human_affordance'] = torch.sigmoid(human_aff['logits'])
            outputs['human_point_features'] = human_aff['features']

        # Object branch
        if compute_object and object_points is not None:
            _, object_encode = self.object_pc_encoder(object_points)
            object_dec = self.object_point_decoder(object_encode)
            object_feat = self.point_feat_proj(object_dec['features'])
            object_feat = self._fuse_semantic(object_feat, E_object, self.object_sem_film)

            object_aff = self.object_affordance(object_feat, E_object)
            outputs['object_affordance_logits'] = object_aff['logits']
            outputs['object_affordance'] = torch.sigmoid(object_aff['logits'])
            outputs['object_point_features'] = object_aff['features']

        return outputs

    def freeze_vlm(self):
        """Freeze VLM parameters."""
        for param in self.vlm.parameters():
            param.requires_grad = False

    def unfreeze_vlm(self):
        """Unfreeze VLM parameters."""
        for param in self.vlm.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> Dict[str, List[nn.Parameter]]:
        """Get grouped trainable parameters for different learning rates."""
        params = {
            'vlm': [p for p in self.vlm.parameters() if p.requires_grad],
            'encoder': [p for p in self.object_pc_encoder.parameters() if p.requires_grad],
            'decoder': (
                list(self.object_point_decoder.parameters()) +
                list(self.point_feat_proj.parameters()) +
                list(self.object_sem_film.parameters())
            ),
            'human_branch': (
                list(self.human_point_provider.parameters()) +
                list(self.human_affordance.parameters())
            ),
            'object_branch': list(self.object_affordance.parameters())
        }
        return params


class DiceLoss(nn.Module):
    """Dice loss for mask/affordance prediction - effective for class imbalance."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: (B, N) predicted probabilities (after sigmoid)
            target: (B, N) ground truth mask
        """
        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class AffordanceLoss(nn.Module):
    """
    Loss function for affordance prediction.
    Supports partial annotations via valid masks.

    Uses Dice + Focal loss combination to handle severe class imbalance
    (contact points are typically only 2-10% of all points).
    Reference: models/losses.py (IVDLoss uses Dice + BCE for masks)
    """

    def __init__(
        self,
        use_focal: bool = True,
        use_dice: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lambda_human: float = 1.0,
        lambda_object: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0
    ):
        super().__init__()
        self.use_focal = use_focal
        self.use_dice = use_dice
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lambda_human = lambda_human
        self.lambda_object = lambda_object
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

        self.dice_loss = DiceLoss(smooth=1.0)

        if not use_focal:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss (same as models/losses.py FocalLoss)."""
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)

        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma
        return (focal_weight * bce).mean()

    def compute_combined_loss(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined Dice + Focal/BCE loss."""
        loss = torch.tensor(0.0, device=pred_logits.device)

        # Focal or BCE loss
        if self.use_focal:
            focal = self.focal_loss(pred_logits, target)
            loss = loss + self.lambda_focal * focal
        else:
            bce = self.bce_loss(pred_logits, target)
            loss = loss + self.lambda_focal * bce

        # Dice loss (operates on probabilities)
        if self.use_dice:
            pred_prob = torch.sigmoid(pred_logits)
            dice = self.dice_loss(pred_prob, target)
            loss = loss + self.lambda_dice * dice

        return loss

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with support for partial annotations.

        Args:
            predictions: Model outputs
            targets: Dictionary with:
                - 'human_mask_gt': (B, 10475) human contact mask
                - 'human_mask_valid': (B,) bool mask for valid human annotations
                - 'object_mask_gt': (B, N) object contact mask
                - 'object_mask_valid': (B,) bool mask for valid object annotations

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        device = next(iter(predictions.values())).device

        # Human affordance loss
        human_loss = torch.tensor(0.0, device=device)
        if 'human_affordance_logits' in predictions:
            human_valid = targets.get('human_mask_valid', None)
            if human_valid is not None and human_valid.any():
                valid_idx = human_valid
                pred_human = predictions['human_affordance_logits'][valid_idx]
                gt_human = targets['human_mask_gt'][valid_idx].float()

                # Align dimensions if needed
                if pred_human.shape[-1] != gt_human.shape[-1]:
                    min_len = min(pred_human.shape[-1], gt_human.shape[-1])
                    pred_human = pred_human[..., :min_len]
                    gt_human = gt_human[..., :min_len]

                human_loss = self.compute_combined_loss(pred_human, gt_human)

        losses['human_loss'] = human_loss

        # Object affordance loss
        object_loss = torch.tensor(0.0, device=device)
        if 'object_affordance_logits' in predictions:
            object_valid = targets.get('object_mask_valid', None)
            if object_valid is not None and object_valid.any():
                valid_idx = object_valid
                pred_object = predictions['object_affordance_logits'][valid_idx]
                gt_object = targets['object_mask_gt'][valid_idx].float()

                # Align dimensions if needed
                if pred_object.shape[-1] != gt_object.shape[-1]:
                    min_len = min(pred_object.shape[-1], gt_object.shape[-1])
                    pred_object = pred_object[..., :min_len]
                    gt_object = gt_object[..., :min_len]

                object_loss = self.compute_combined_loss(pred_object, gt_object)

        losses['object_loss'] = object_loss

        # Total loss
        losses['total_loss'] = (
            self.lambda_human * human_loss +
            self.lambda_object * object_loss
        )

        return losses


def build_affordance_model(config: dict) -> UnifiedAffordanceModel:
    """Build Unified Affordance Model from configuration."""
    model = UnifiedAffordanceModel(
        d_tr=config.get('d_tr', 256),
        num_human_points=config.get('num_human_points', 10475),
        use_lightweight_vlm=config.get('use_lightweight_vlm', True),
        vlm_model_name=config.get('vlm_model_name', "llava-hf/llava-1.5-7b-hf"),
        lora_r=config.get('lora_r', 16),
        lora_alpha=config.get('lora_alpha', 32),
        freeze_vlm=config.get('freeze_vlm', True),
        dropout=config.get('dropout', 0.3),
        device=config.get('device', 'cuda')
    )
    return model
