"""
Affordance Pre-training Module

Note: Imports are done lazily to avoid circular import issues
with the root models/ directory.
"""

__all__ = [
    # Models
    'UnifiedAffordanceModel',
    'AffordanceLoss',
    'build_affordance_model',

    # Datasets
    'HumanAffordanceDataset',
    'ObjectAffordanceDataset',
    'InterCapAffordanceDataset',
    'AffordanceTransform',
    'create_affordance_dataloaders',

    # Weight loading
    'load_affordance_to_main',
    'create_main_model_with_pretrained_affordance',
    'verify_weight_loading'
]


def __getattr__(name):
    """Lazy import to avoid circular dependency with root models/ directory."""
    # Model imports
    if name in ('UnifiedAffordanceModel', 'AffordanceLoss', 'build_affordance_model'):
        from .aff_models.unified_aff_model import UnifiedAffordanceModel, AffordanceLoss, build_affordance_model
        return {
            'UnifiedAffordanceModel': UnifiedAffordanceModel,
            'AffordanceLoss': AffordanceLoss,
            'build_affordance_model': build_affordance_model
        }[name]

    # Dataset imports
    if name in ('HumanAffordanceDataset', 'ObjectAffordanceDataset', 'InterCapAffordanceDataset',
                'AffordanceTransform', 'create_affordance_dataloaders'):
        from .data.unified_aff_dataset import (
            HumanAffordanceDataset, ObjectAffordanceDataset, InterCapAffordanceDataset,
            AffordanceTransform, create_affordance_dataloaders
        )
        return {
            'HumanAffordanceDataset': HumanAffordanceDataset,
            'ObjectAffordanceDataset': ObjectAffordanceDataset,
            'InterCapAffordanceDataset': InterCapAffordanceDataset,
            'AffordanceTransform': AffordanceTransform,
            'create_affordance_dataloaders': create_affordance_dataloaders
        }[name]

    # Weight loading imports
    if name in ('load_affordance_to_main', 'create_main_model_with_pretrained_affordance', 'verify_weight_loading'):
        from .load_to_main import (
            load_affordance_to_main, create_main_model_with_pretrained_affordance, verify_weight_loading
        )
        return {
            'load_affordance_to_main': load_affordance_to_main,
            'create_main_model_with_pretrained_affordance': create_main_model_with_pretrained_affordance,
            'verify_weight_loading': verify_weight_loading
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
