# InterActVLM-Discrete (IVD)

Discrete Human-Object Interaction Contact Prediction from Single RGB Image

## Overview

IVD predicts discrete human-object interaction (HOI) contacts from a single RGB image:

- **Human Output**: Binary classification (contact or not) for 87 predefined anatomical points on the SMPL-X body mesh (including hands, face, and knees)
- **Object Output**: 3D coordinate regression $(x, y, z)$ for contact locations on the object surface

## Architecture

The model consists of 5 main stages:

### Stage 1: Semantic Reasoning (VLM Module)
- Uses LLaVA-v1.5 with LoRA fine-tuning
- Extracts semantic embeddings `E_human` and `E_object` from specialized tokens `<HCON>` and `<OCON>`
- Input: RGB Image (B, 3, 224, 224)
- Output: (B, D_tr) embeddings for human and object

### Stage 2: Multi-View Visual Encoding
- DINOv2 (ViT-Small) encoder for multi-view rendered images
- Input: Rendered views (B*J, 3, 256, 256) from J=4 viewpoints
- Output: Feature maps (B*J, D_tr, 16, 16)

### Stage 3: Branching Expert Decoders
- Human Branch: Predicts 2D human contact mask + enhanced features
- Object Branch: Predicts 2D object affordance mask + enhanced features
- Auxiliary supervision for geometry understanding

### Stage 4: Shared Interaction Transformer
- Cross-attention: Queries attend to multi-view visual features
- Self-attention: Body and object queries interact for physical reasoning
- 87 learnable body queries + K object queries

### Stage 5: Prediction Heads
- Human Contact Head: MLP (D_tr → 1) → (B, 87) binary predictions
- Object Coordinate Head: MLP (D_tr → 3) → (B, K, 3) 3D coordinates

## Installation

```bash
cd InterActVLM-point

# 创建虚拟环境
uv venv --python 3.10
source .venv/bin/activate  # Linux/Mac

# 方法1: 使用 uv sync (推荐，已配置阿里云镜像)
uv sync

# 方法2: 使用 requirements.txt
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### PyTorch3D 安装 (可选，用于多视图渲染)

PyTorch3D 在 Linux 上没有预编译 wheel，需要从源码安装：

```bash
# 确保已安装 CUDA 和 PyTorch
# 方法1: 从 GitHub 安装
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" -i https://mirrors.aliyun.com/pypi/simple/

# 方法2: 从源码编译
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

### 镜像配置

项目已在 `pyproject.toml` 中配置阿里云镜像。也可以全局配置：

```bash
# 创建或编辑 ~/.config/uv/uv.toml
mkdir -p ~/.config/uv
cat > ~/.config/uv/uv.toml << 'EOF'
index-url = "https://mirrors.aliyun.com/pypi/simple/"
EOF
```

## Data Structure

```
data/
├── images/           # Original RGB scene images
│   ├── 00001.jpg
│   └── ...
├── human/            # Human mesh data
│   ├── 00001.npz     # vertices, faces, (contacts)
│   └── ...
├── object/           # Object point cloud data
│   ├── 00001.npz     # points, (contact_coords)
│   └── ...
├── annotations/      # Contact annotations
│   ├── 00001.json    # human_contact_labels, object_contact_coords
│   └── ...
├── renders/          # Pre-rendered multi-view images (optional)
│   └── ...
└── splits/           # Train/val/test splits
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### Annotation Format

```json
{
    "human_contact_labels": [0, 1, 0, ...],  // 87 binary labels
    "object_contact_coords": [[x1, y1, z1], [x2, y2, z2], ...]  // K coordinates
}
```

## 87 Body Contact Points

The model uses 87 predefined anatomical keypoints on the SMPL-X mesh, including:

- Hands (left/right): palm, back, thumb, index, middle, ring, pinky
- Arms (left/right): upper arm (front, back, up, down), forearm (back, wrist, thumb, pinky)
- Legs (left/right): upper leg (front, back, inner, outer), lower leg (front, back, inner, outer)
- Feet (left/right): toe base, instep, sole
- Knees (left/right): front and back regions
- Torso: spine regions, hip regions, buttocks
- Neck & Shoulders: front and back regions
- Face: cheeks, chin, head top, mouth

## Training

### Basic Training

```bash
python train.py --config configs/default.yaml --data_root ./data
```

### Training with W&B Logging

```bash
python train.py --config configs/default.yaml --data_root ./data --wandb
```

### Resume from Checkpoint

```bash
python train.py --config configs/default.yaml --checkpoint checkpoints/latest.pth
```

### Training Phases

1. **Phase 1** (epochs 1-20): Train with auxiliary mask loss to learn geometry
2. **Phase 2** (epochs 21+): Full training with contact classification and regression

## Inference

### Single Image

```bash
python inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --image path/to/image.jpg \
    --output ./outputs \
    --visualize
```

### Batch Inference

```bash
python inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best.pth \
    --data_dir path/to/images \
    --output ./outputs
```

### Python API

```python
from inference import IVDPredictor

# Initialize predictor
predictor = IVDPredictor(
    config_path='configs/default.yaml',
    checkpoint_path='checkpoints/best.pth',
    keypoints_json='data/part_kp.json',
    device='cuda'
)

# Run prediction
result = predictor.predict('image.jpg', threshold=0.5)

print(f"Contacts: {result['num_contacts']}/87")
print(f"Contact regions: {result['human_contact_names']}")
print(f"Object coords: {result['object_coords']}")
```

## Loss Functions

1. **L_human**: Binary Cross Entropy (or Focal Loss) for 74 body points
2. **L_object**: L2 distance or Chamfer Loss for 3D coordinates
3. **L_aux_mask**: Dice + BCE Loss for 2D auxiliary masks

Total Loss: `L = λ₁·L_human + λ₂·L_object + λ₃·L_aux_mask`

## Configuration

Key configuration parameters in `configs/default.yaml`:

```yaml
model:
  d_tr: 256          # Transformer dimension
  num_body_points: 74
  num_obj_queries: 4  # K object contact queries
  num_views: 4        # J rendered views

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1.0e-4
  lambda_human: 1.0
  lambda_object: 1.0
  lambda_aux_mask: 0.5
```

## Model Variants

### Full Model (IVDModel)
- Uses VLM for semantic reasoning
- Best accuracy, higher memory requirements

### Lite Model (IVDModelLite)
- Removes VLM, uses learned embeddings
- Faster inference, lower memory

## Tensor Shapes Reference

| Component | Tensor | Shape |
|:---|:---|:---|
| Input RGB | `img_rgb` | (B, 3, 224, 224) |
| Render Views | `render_views` | (B·J, 3, 256, 256) |
| VLM Embeddings | `emb_h`, `emb_o` | (B, D_tr) |
| Encoder Features | `feat_base` | (B·J, D_tr, 16, 16) |
| Auxiliary Masks | `mask_h`, `mask_o` | (B, J, 1, 256, 256) |
| Branch Features | `feat_h`, `feat_o` | (B, J, D_tr, 16, 16) |
| Human Queries | `queries_h` | (B, 74, D_tr) |
| Object Queries | `queries_o` | (B, K, D_tr) |
| Human Output | `pred_h` | (B, 74) |
| Object Output | `pred_o` | (B, K, 3) |

## Evaluation Metrics

### Human Contact
- Accuracy, Precision, Recall, F1-score
- Per-body-part metrics
- AUC-ROC, Average Precision

### Object Coordinates
- Mean L2 distance
- Chamfer distance
- Accuracy at 5cm, 10cm, 20cm thresholds

## Project Structure

```
InterActVLM-point/
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── ivd_model.py
│   ├── vlm_module.py
│   ├── dino_encoder.py
│   ├── branch_decoder.py
│   ├── interaction_transformer.py
│   └── losses.py
├── utils/
│   ├── __init__.py
│   ├── keypoints.py
│   ├── renderer.py
│   └── metrics.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{ivd2024,
  title={InterActVLM-Discrete: Discrete Human-Object Interaction Contact Prediction},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
