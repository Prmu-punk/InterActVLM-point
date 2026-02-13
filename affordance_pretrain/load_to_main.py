"""
Load pre-trained affordance weights to main IVDModel
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models import build_model


# Module mappings from UnifiedAffordanceModel to IVDModel
# These should be identical, but listed here for clarity
AFFORDANCE_MODULES = {
    # Shared modules
    'vlm': 'vlm',
    'object_pc_encoder': 'object_pc_encoder',
    'object_point_decoder': 'object_point_decoder',
    'point_feat_proj': 'point_feat_proj',
    'object_sem_film': 'object_sem_film',

    # Human branch
    'human_point_provider': 'human_point_provider',
    'human_affordance': 'human_affordance',

    # Object branch
    'object_affordance': 'object_affordance'
}


def load_affordance_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load affordance pre-training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt


def get_matching_keys(
    affordance_state: Dict[str, torch.Tensor],
    main_state: Dict[str, torch.Tensor]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Find matching keys between affordance and main model.

    Returns:
        matched: Keys that match in both models
        missing_in_main: Keys in affordance but not in main
        missing_in_aff: Keys in main but not in affordance
    """
    aff_keys = set(affordance_state.keys())
    main_keys = set(main_state.keys())

    matched = aff_keys & main_keys
    missing_in_main = aff_keys - main_keys
    missing_in_aff = main_keys - aff_keys

    return sorted(matched), sorted(missing_in_main), sorted(missing_in_aff)


def load_affordance_to_main(
    main_model,
    affordance_checkpoint: str,
    modules_to_load: Optional[List[str]] = None,
    strict: bool = False,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load pre-trained affordance weights into main IVDModel.

    Args:
        main_model: IVDModel instance
        affordance_checkpoint: Path to affordance pre-training checkpoint
        modules_to_load: List of module prefixes to load (default: all)
                        e.g., ['human_affordance', 'object_affordance']
        strict: If True, raise error on missing keys
        verbose: Print loading information

    Returns:
        loaded_keys: List of keys that were loaded
        skipped_keys: List of keys that were skipped
    """
    # Load checkpoint
    aff_state = load_affordance_checkpoint(affordance_checkpoint)
    main_state = main_model.state_dict()

    # Default: load all matching modules
    if modules_to_load is None:
        modules_to_load = list(AFFORDANCE_MODULES.keys())

    # Find matching keys
    matched, missing_in_main, _ = get_matching_keys(aff_state, main_state)

    if verbose:
        print(f"Affordance checkpoint: {len(aff_state)} keys")
        print(f"Main model: {len(main_state)} keys")
        print(f"Matching keys: {len(matched)}")
        if missing_in_main:
            print(f"Keys in affordance but not in main: {len(missing_in_main)}")

    # Filter keys by modules_to_load
    keys_to_load = []
    for key in matched:
        module_prefix = key.split('.')[0]
        if module_prefix in modules_to_load:
            keys_to_load.append(key)

    if verbose:
        print(f"Keys to load (filtered): {len(keys_to_load)}")

    # Check shape compatibility
    loaded_keys = []
    skipped_keys = []

    for key in keys_to_load:
        aff_shape = aff_state[key].shape
        main_shape = main_state[key].shape

        if aff_shape == main_shape:
            main_state[key].copy_(aff_state[key])
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
            if verbose:
                print(f"Shape mismatch for {key}: aff={aff_shape}, main={main_shape}")

    # Load state dict
    main_model.load_state_dict(main_state)

    if verbose:
        print(f"\nLoaded {len(loaded_keys)} keys successfully")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys due to shape mismatch")

        # Print module summary
        module_counts = {}
        for key in loaded_keys:
            module = key.split('.')[0]
            module_counts[module] = module_counts.get(module, 0) + 1

        print("\nLoaded modules:")
        for module, count in sorted(module_counts.items()):
            print(f"  {module}: {count} parameters")

    if strict and skipped_keys:
        raise RuntimeError(f"Strict mode: {len(skipped_keys)} keys could not be loaded")

    return loaded_keys, skipped_keys


def create_main_model_with_pretrained_affordance(
    main_model_config: dict,
    affordance_checkpoint: str,
    modules_to_load: Optional[List[str]] = None,
    device: str = 'cuda'
):
    """
    Create main IVDModel with pre-trained affordance weights.

    Args:
        main_model_config: Configuration for IVDModel
        affordance_checkpoint: Path to affordance checkpoint
        modules_to_load: Modules to load from affordance checkpoint
        device: Device to load model on

    Returns:
        IVDModel with loaded weights
    """
    # Build main model
    main_model = build_model(main_model_config)

    # Load affordance weights
    load_affordance_to_main(
        main_model,
        affordance_checkpoint,
        modules_to_load=modules_to_load,
        verbose=True
    )

    return main_model.to(device)


def verify_weight_loading(
    main_model,
    affordance_checkpoint: str,
    modules_to_verify: Optional[List[str]] = None
) -> bool:
    """
    Verify that weights were loaded correctly.

    Args:
        main_model: IVDModel with loaded weights
        affordance_checkpoint: Path to affordance checkpoint
        modules_to_verify: Modules to verify

    Returns:
        True if all weights match
    """
    aff_state = load_affordance_checkpoint(affordance_checkpoint)
    main_state = main_model.state_dict()

    if modules_to_verify is None:
        modules_to_verify = list(AFFORDANCE_MODULES.keys())

    all_match = True
    for key in aff_state:
        module_prefix = key.split('.')[0]
        if module_prefix not in modules_to_verify:
            continue

        if key in main_state:
            if aff_state[key].shape == main_state[key].shape:
                if not torch.allclose(aff_state[key], main_state[key].cpu()):
                    print(f"Mismatch: {key}")
                    all_match = False
            else:
                print(f"Shape mismatch: {key}")
                all_match = False

    if all_match:
        print("All weights verified successfully!")
    return all_match


def main():
    parser = argparse.ArgumentParser(description="Load affordance weights to main model")
    parser.add_argument('--affordance_ckpt', type=str, required=True,
                        help="Path to affordance pre-training checkpoint")
    parser.add_argument('--output', type=str, required=True,
                        help="Output path for main model with loaded weights")

    # Model config
    parser.add_argument('--d_tr', type=int, default=256)
    parser.add_argument('--num_body_points', type=int, default=87)
    parser.add_argument('--num_object_queries', type=int, default=87)
    parser.add_argument('--use_lightweight_vlm', action='store_true')

    # Loading options
    parser.add_argument('--modules', type=str, nargs='+', default=None,
                        help="Specific modules to load (default: all)")
    parser.add_argument('--strict', action='store_true',
                        help="Fail if any keys cannot be loaded")
    parser.add_argument('--verify', action='store_true',
                        help="Verify weights after loading")

    args = parser.parse_args()

    # Build main model config
    main_config = {
        'd_tr': args.d_tr,
        'num_body_points': args.num_body_points,
        'num_object_queries': args.num_object_queries,
        'use_lightweight_vlm': args.use_lightweight_vlm,
        'device': 'cpu'
    }

    # Build main model
    print("Building main model...")
    main_model = build_model(main_config)

    # Load affordance weights
    print(f"\nLoading affordance weights from: {args.affordance_ckpt}")
    loaded, skipped = load_affordance_to_main(
        main_model,
        args.affordance_ckpt,
        modules_to_load=args.modules,
        strict=args.strict
    )

    # Verify if requested
    if args.verify:
        print("\nVerifying loaded weights...")
        verify_weight_loading(main_model, args.affordance_ckpt, args.modules)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': main_model.state_dict(),
        'affordance_checkpoint': args.affordance_ckpt,
        'loaded_keys': loaded,
        'config': main_config
    }, output_path)

    print(f"\nSaved main model with affordance weights to: {output_path}")


if __name__ == '__main__':
    main()
