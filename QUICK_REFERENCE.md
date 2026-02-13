# InterActVLM-point 快速参考卡

## 1️⃣ 关键点数量更新完成 (74 → 87)

所有配置、代码和文档已更新以支持 87 个 SMPL-X 关键点。

### 受影响的文件
- ✅ `configs/default.yaml` - 主配置
- ✅ `data/dataset.py` - 数据加载器默认参数
- ✅ `README.md` - 所有文档
- ✅ `utils/keypoints.py` - 工具类
- ✅ `utils/metrics.py` - 评估指标

### 新增 13 个关键点
- Face: `Chin`, `Head_top`, `Mouth`, `leftCheek`, `rightCheek`
- Knees: `leftKnee_front`, `leftKnee_back`, `rightKnee_front`, `rightKnee_back`
- Elbows: `leftElbow_front`, `leftElbow_back`, `rightElbow_front`, `rightElbow_back`

---

## 2️⃣ W&B 登陆问题解决方案

### 快速修复 (推荐 ⭐)

```bash
# 步骤 1: 登出当前用户
wandb logout

# 步骤 2: 登陆你的账户
wandb login
# 输入你的 API Key (从 https://wandb.ai/authorize 获取)

# 步骤 3: 验证
wandb whoami

# 步骤 4: 启用 W&B 训练
python train.py --config configs/default.yaml --data_root ./data --wandb
```

### 备选方案

#### 方案 A: 离线模式 (本地日志)
```bash
export WANDB_MODE=offline
python train.py --config configs/default.yaml --data_root ./data --wandb
```

#### 方案 B: 禁用 W&B (只保存本地日志)
```bash
python train.py --config configs/default.yaml --data_root ./data
# (不添加 --wandb 标志)
```

---

## 3️⃣ 查看训练曲线

### 方式 1: W&B 在线仪表板 (实时，最佳) ⭐

```bash
# 训练期间在线发送数据
wandb login
python train.py --config configs/default.yaml --data_root ./data --wandb

# 访问: https://wandb.ai/your-username/InterActVLM-Discrete
```

**优点**:
- 实时查看训练进度
- 对比多个运行
- 远程查看 (任何设备)

### 方式 2: TensorBoard (本地，轻量级)

```bash
# 修改 configs/default.yaml 启用 TensorBoard
# logging:
#   use_tensorboard: true

tensorboard --logdir ./logs/
# 访问: http://localhost:6006
```

### 方式 3: 离线 W&B 日志 (本地)

```bash
# 查看最新运行
ls -ltr ./wandb/ | tail -5

# 查看汇总指标
cat ./wandb/latest-run/run-*/files/summary.json | python -m json.tool
```

### 方式 4: 命令行查看

```bash
# 查看实时日志
tail -f ./logs/train_*.log

# 提取损失值
grep "Loss" ./logs/train_*.log | tail -20
```

### 推荐使用场景

| 场景 | 推荐方式 | 命令 |
|------|--------|------|
| 本地开发，快速反馈 | TensorBoard | `tensorboard --logdir ./logs/` |
| 正式训练，详细分析 | W&B 在线 | `wandb login && python train.py ... --wandb` |
| 服务器训练，看日志 | 离线 W&B | `export WANDB_MODE=offline && python train.py ... --wandb` |
| 简单记录，不需可视化 | 禁用 W&B | `python train.py ... ` (无 --wandb) |

---

## 4️⃣ 常用命令速查

```bash
# 训练 (最完整配置)
python train.py \
  --config configs/default.yaml \
  --data_root ./data \
  --wandb \
  --checkpoint checkpoints/latest.pth

# 从检查点恢复
python train.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth

# 推理
python inference.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --image ./test.jpg \
  --output ./outputs \
  --visualize

# 批处理推理
python inference.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_dir ./data/test/images \
  --output ./outputs
```

---

## 5️⃣ 文件结构参考

```
InterActVLM-point/
├── configs/
│   └── default.yaml         # ← 87 num_body_points
├── models/
│   ├── ivd_model.py         # ← 默认 num_body_points=87
│   └── ...
├── data/
│   ├── dataset.py           # ← 87 num_object_queries
│   ├── part_kp.json         # ← 87 关键点定义
│   └── ...
├── utils/
│   ├── keypoints.py         # ← 87 关键点文档
│   ├── metrics.py           # ← 87 标签文档
│   └── ...
├── train.py                 # ← 使用 --wandb 启用 W&B
├── inference.py
├── README.md                # ← 已更新为 87
├── WANDB_SETUP.md           # ← 新增指南 ⭐
├── logs/                    # ← 训练日志
├── checkpoints/             # ← 模型权重
├── wandb/                   # ← 离线 W&B 日志
└── outputs/                 # ← 推理结果
```

---

## ℹ️ 更多信息

- **W&B 完整指南**: 见 `WANDB_SETUP.md`
- **官方 W&B 文档**: https://docs.wandb.ai/
- **离线模式**: https://docs.wandb.ai/guides/offline
- **API Key 获取**: https://wandb.ai/authorize

---

## 🔧 故障排除

### Q: 训练时 W&B 错误？
**A**:
```bash
wandb offline  # 临时禁用
# 或
export WANDB_MODE=offline
```

### Q: 无法登陆 W&B？
**A**:
```bash
wandb login --relogin
# 或从 https://wandb.ai/authorize 获取新的 API Key
```

### Q: 想看其他用户的 W&B 数据？
**A**:
```bash
wandb sync ./wandb/offline-run-*/ --project InterActVLM-Discrete
```

---

**最后更新**: 2026-02-07
**状态**: ✅ 完整
