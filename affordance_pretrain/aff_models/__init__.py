"""
Affordance Pre-training Models

Note: Imports are done lazily to avoid circular import issues
with the root models/ directory.
"""

__all__ = [
    'UnifiedAffordanceModel',
    'AffordanceLoss',
    'build_affordance_model'
]


def __getattr__(name):
    """Lazy import to avoid circular dependency with root models/ directory."""
    if name in ('UnifiedAffordanceModel', 'AffordanceLoss', 'build_affordance_model'):
        from .unified_aff_model import UnifiedAffordanceModel, AffordanceLoss, build_affordance_model
        return {
            'UnifiedAffordanceModel': UnifiedAffordanceModel,
            'AffordanceLoss': AffordanceLoss,
            'build_affordance_model': build_affordance_model
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
