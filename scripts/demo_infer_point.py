import os
import numpy as np
import torch
from PIL import Image

from models import build_model


def load_rgb(image_path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def load_object_points(points_path: str, device: torch.device) -> torch.Tensor:
    if os.path.exists(points_path):
        pts = np.load(points_path).astype(np.float32)
        if pts.ndim == 2:
            pts = pts[None, ...]
    else:
        # Fallback: random point cloud if file is missing
        pts = np.random.randn(1, 1024, 3).astype(np.float32)
    return torch.from_numpy(pts).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model entrypoint: build_model -> IVDModel
    model = build_model({
        "d_tr": 256,
        "num_body_points": 87,
        "num_object_queries": 87,
        "use_lightweight_vlm": True,
        "device": str(device),
    }).to(device)
    model.eval()

    rgb_image = load_rgb("data/images/example.jpg", device)
    object_points = load_object_points("data/object_points_example.npy", device)

    with torch.no_grad():
        outputs = model(rgb_image, object_points, return_aux=False)

    print("human_contact:", outputs["human_contact"].shape)         # (B, 87)
    print("object_index_logits:", outputs["object_index_logits"].shape)  # (B, 87, 1024)
    print("human_affordance:", outputs["human_affordance"].shape)   # (B, 10475)
    print("object_affordance:", outputs["object_affordance"].shape) # (B, 1024)


if __name__ == "__main__":
    main()
