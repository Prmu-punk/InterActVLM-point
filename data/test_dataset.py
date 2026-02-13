#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from data.dataset import IVDDataset
from data.transforms import get_val_transforms


_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


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


def _to_uint8_image(
    tensor: torch.Tensor,
    unnormalize: bool = False,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std: Tuple[float, float, float] = _IMAGENET_STD,
) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W) tensor, got {tuple(tensor.shape)}")

    img = tensor.detach().cpu().float()

    if unnormalize:
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        img = img * std_t + mean_t

    img = img.clamp(0.0, 1.0)

    # If still outside [0,1], normalize heuristically
    max_val = float(img.max()) if img.numel() else 0.0
    if max_val > 1.0:
        if max_val <= 255.0:
            img = img / 255.0
        else:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    img = (img * 255.0).byte().permute(1, 2, 0).numpy()
    return img


def _save_render_views(
    render_views: Any,
    out_dir: Path,
    sample_id: str,
    unnormalize: bool,
) -> None:
    if isinstance(render_views, np.ndarray):
        render_views = torch.from_numpy(render_views)
    if not isinstance(render_views, torch.Tensor):
        raise TypeError(f"Unsupported render_views type: {type(render_views)}")

    if render_views.ndim != 4:
        raise ValueError(f"Expected (V, 3, H, W) renders, got {tuple(render_views.shape)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(render_views.shape[0]):
        img = _to_uint8_image(render_views[i], unnormalize=unnormalize)
        Image.fromarray(img).save(out_dir / f"{sample_id}_view_{i:02d}.png")


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
    parser.add_argument("--save-renders", action="store_true", help="Save render views as PNGs")
    parser.add_argument("--out-dir", default="render_debug", help="Output directory for renders")
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

    sample_id = str(sample.get("sample_id"))
    out_dir = Path(args.out_dir)
    _save_render_views(
        sample["render_views"],
        out_dir=out_dir,
        sample_id=sample_id,
        unnormalize=args.use_transform,
    )
    print(f"Saved render views to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
