#!/usr/bin/env python3
import argparse
import json
from typing import Any, Dict

import numpy as np
import torch

from data.dataset import IVDDataset
from data.transforms import get_val_transforms


def _summarize_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "min": value.min().item() if value.numel() else None,
            "max": value.max().item() if value.numel() else None,
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "np.ndarray",
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "min": float(value.min()) if value.size else None,
            "max": float(value.max()) if value.size else None,
        }
    return {"type": type(value).__name__, "value": value}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick sanity check for IVDDataset.__getitem__")
    parser.add_argument("--intercap-data", required=True, help="Dataset root directory")
    parser.add_argument("--annot-data", required=True, help="Annotation directory")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--idx", type=int, default=0, help="Sample index to inspect")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-object-points", type=int, default=1024)
    parser.add_argument("--num-object-queries", type=int, default=4)
    parser.add_argument("--use-transform", action="store_true", help="Apply val transforms")
    parser.add_argument("--render-on-fly", action="store_true", help="Try on-the-fly renders")
    args = parser.parse_args()

    transform = None
    if args.use_transform:
        transform = get_val_transforms(image_size=224, render_size=256)

    dataset = IVDDataset(
        intercap_data=args.intercap_data,
        annot_data=args.annot_data,
        split=args.split,
        transform=transform,
        num_views=args.num_views,
        render_on_fly=args.render_on_fly,
        num_object_points=args.num_object_points,
        num_object_queries=args.num_object_queries,
    )

    if len(dataset) == 0:
        raise SystemExit("Dataset is empty for this split. Check split file or paths.")

    if args.idx < 0 or args.idx >= len(dataset):
        raise SystemExit(f"idx out of range: 0..{len(dataset) - 1}")

    sample = dataset[args.idx]
    print(f"sample_id: {sample.get('sample_id')}")
    summary = {k: _summarize_value(v) for k, v in sample.items() if k != "sample_id"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
