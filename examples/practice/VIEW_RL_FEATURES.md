# 如何查看强化学习使用的特征值

本指南展示如何查看 LeRobot 中用于强化学习的各种特征值。

## 📋 目录

1. [查看数据集特征](#查看数据集特征)
2. [查看策略的输入输出特征](#查看策略的输入输出特征)
3. [查看特征类型](#查看特征类型)
4. [完整示例代码](#完整示例代码)

---

## 查看数据集特征

### 方法 1：从数据集元数据查看

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from pprint import pprint

# 加载数据集元数据（不需要下载完整数据集）
repo_id = "lerobot/pusht"
ds_meta = LeRobotDatasetMetadata(repo_id)

# 查看所有特征
print("=== 数据集特征 ===")
pprint(ds_meta.features)

# 查看特定类型的特征
print("\n=== 观察特征（STATE） ===")
for key, feature in ds_meta.features.items():
    if key.startswith("observation"):
        print(f"{key}: {feature}")

print("\n=== 动作特征（ACTION） ===")
for key, feature in ds_meta.features.items():
    if key.startswith("action"):
        print(f"{key}: {feature}")

print("\n=== 奖励特征（REWARD） ===")
if "reward" in ds_meta.features:
    print(f"reward: {ds_meta.features['reward']}")
```

### 方法 2：从已加载的数据集查看

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("lerobot/pusht")

# 查看特征
print("=== 数据集特征 ===")
print(dataset.features)  # 或 dataset.meta.features

# 查看单个样本，了解实际数据
sample = dataset[0]
print("\n=== 样本数据键 ===")
for key in sample.keys():
    print(f"{key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
```

---

## 查看策略的输入输出特征

### 方法 1：从策略配置查看

```python
from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

# 加载数据集元数据
ds_meta = LeRobotDatasetMetadata("lerobot/pusht")

# 创建策略（会自动从数据集推断特征）
policy = make_policy(
    cfg=...,  # 你的策略配置
    ds_meta=ds_meta
)

# 查看输入特征
print("=== 策略输入特征 ===")
for key, feature in policy.config.input_features.items():
    print(f"{key}:")
    print(f"  - 类型: {feature.type}")
    print(f"  - 形状: {feature.shape}")

# 查看输出特征
print("\n=== 策略输出特征 ===")
for key, feature in policy.config.output_features.items():
    print(f"{key}:")
    print(f"  - 类型: {feature.type}")
    print(f"  - 形状: {feature.shape}")
```

### 方法 2：从预训练模型查看

```python
from lerobot.policies.factory import make_policy

# 加载预训练模型
policy = make_policy(
    cfg=...,
    pretrained_path="lerobot/diffusion_pusht"
)

# 查看配置
print("=== 输入特征 ===")
pprint(policy.config.input_features)

print("\n=== 输出特征 ===")
pprint(policy.config.output_features)
```

---

## 查看特征类型

LeRobot 定义了以下特征类型：

```python
from lerobot.configs.types import FeatureType

# 特征类型枚举
print("=== 特征类型 ===")
print(f"STATE: {FeatureType.STATE}")      # 状态特征（如关节位置）
print(f"VISUAL: {FeatureType.VISUAL}")    # 视觉特征（图像）
print(f"ACTION: {FeatureType.ACTION}")    # 动作特征
print(f"REWARD: {FeatureType.REWARD}")    # 奖励特征
print(f"LANGUAGE: {FeatureType.LANGUAGE}") # 语言特征
print(f"ENV: {FeatureType.ENV}")         # 环境特征
```

### 按类型分类特征

```python
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

# 将数据集特征转换为策略特征
ds_meta = LeRobotDatasetMetadata("lerobot/pusht")
policy_features = dataset_to_policy_features(ds_meta.features)

# 按类型分类
features_by_type = {
    FeatureType.STATE: [],
    FeatureType.VISUAL: [],
    FeatureType.ACTION: [],
    FeatureType.REWARD: [],
}

for key, feature in policy_features.items():
    features_by_type[feature.type].append((key, feature))

# 打印分类结果
for feature_type, features in features_by_type.items():
    if features:
        print(f"\n=== {feature_type.value} 特征 ===")
        for key, feature in features:
            print(f"  {key}: shape={feature.shape}")
```

---

## 完整示例代码

### 示例 1：查看数据集的所有特征

```python
#!/usr/bin/env python3
"""查看数据集特征示例"""

from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

def view_dataset_features(repo_id: str):
    """查看数据集的所有特征"""
    
    # 加载元数据
    ds_meta = LeRobotDatasetMetadata(repo_id)
    
    print(f"数据集: {repo_id}")
    print(f"总帧数: {ds_meta.total_frames}")
    print(f"总回合数: {ds_meta.total_episodes}")
    print(f"FPS: {ds_meta.fps}")
    
    print("\n" + "="*60)
    print("原始数据集特征")
    print("="*60)
    pprint(ds_meta.features)
    
    # 转换为策略特征
    policy_features = dataset_to_policy_features(ds_meta.features)
    
    print("\n" + "="*60)
    print("策略特征（按类型分类）")
    print("="*60)
    
    # 按类型分类
    by_type = {}
    for key, feature in policy_features.items():
        if feature.type not in by_type:
            by_type[feature.type] = []
        by_type[feature.type].append((key, feature))
    
    # 打印
    for feature_type in [FeatureType.STATE, FeatureType.VISUAL, FeatureType.ACTION, FeatureType.REWARD]:
        if feature_type in by_type:
            print(f"\n【{feature_type.value}】")
            for key, feature in by_type[feature_type]:
                print(f"  {key}")
                print(f"    形状: {feature.shape}")
                print(f"    类型: {feature.type}")

if __name__ == "__main__":
    # 查看 PushT 数据集
    view_dataset_features("lerobot/pusht")
```

### 示例 2：查看策略的输入输出特征

```python
#!/usr/bin/env python3
"""查看策略特征示例"""

from lerobot.policies.factory import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

def view_policy_features(repo_id: str, policy_type: str = "diffusion"):
    """查看策略的输入输出特征"""
    
    # 加载数据集元数据
    ds_meta = LeRobotDatasetMetadata(repo_id)
    
    # 转换为策略特征
    features = dataset_to_policy_features(ds_meta.features)
    
    # 分离输入和输出特征
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    
    print(f"数据集: {repo_id}")
    print(f"策略类型: {policy_type}")
    
    print("\n" + "="*60)
    print("输入特征（观察）")
    print("="*60)
    for key, feature in input_features.items():
        print(f"\n{key}:")
        print(f"  类型: {feature.type.value}")
        print(f"  形状: {feature.shape}")
    
    print("\n" + "="*60)
    print("输出特征（动作）")
    print("="*60)
    for key, feature in output_features.items():
        print(f"\n{key}:")
        print(f"  类型: {feature.type.value}")
        print(f"  形状: {feature.shape}")

if __name__ == "__main__":
    view_policy_features("lerobot/pusht", "diffusion")
```

### 示例 3：查看强化学习特定特征（State, Action, Reward）

```python
#!/usr/bin/env python3
"""查看强化学习特征示例"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

def view_rl_features(repo_id: str):
    """查看强化学习相关的特征（State, Action, Reward）"""
    
    # 加载数据集
    dataset = LeRobotDataset(repo_id, episodes=[0])  # 只加载第一个回合
    
    # 获取一个样本
    sample = dataset[0]
    
    print(f"数据集: {repo_id}")
    print("\n" + "="*60)
    print("强化学习特征")
    print("="*60)
    
    # State（状态/观察）
    print("\n【STATE - 状态特征】")
    state_keys = [k for k in sample.keys() if k.startswith("observation")]
    for key in state_keys:
        value = sample[key]
        print(f"  {key}:")
        print(f"    形状: {value.shape}")
        print(f"    数据类型: {value.dtype}")
        if value.numel() < 20:  # 如果元素少，打印值
            print(f"    值: {value}")
    
    # Action（动作）
    print("\n【ACTION - 动作特征】")
    action_keys = [k for k in sample.keys() if k.startswith("action")]
    for key in action_keys:
        value = sample[key]
        print(f"  {key}:")
        print(f"    形状: {value.shape}")
        print(f"    数据类型: {value.dtype}")
        if value.numel() < 20:
            print(f"    值: {value}")
    
    # Reward（奖励）
    print("\n【REWARD - 奖励特征】")
    if "reward" in sample:
        reward = sample["reward"]
        print(f"  reward:")
        print(f"    形状: {reward.shape}")
        print(f"    数据类型: {reward.dtype}")
        print(f"    值: {reward.item()}")
    else:
        print("  （此数据集不包含奖励信息）")
    
    # Done（回合结束标志）
    print("\n【DONE - 回合结束标志】")
    if "done" in sample:
        done = sample["done"]
        print(f"  done: {done.item()}")
    else:
        print("  （此数据集不包含 done 标志）")

if __name__ == "__main__":
    view_rl_features("lerobot/pusht")
```

---

## 常用特征键名

### 观察（Observation）特征

- `observation.state` - 状态向量（如关节位置、速度等）
- `observation.images.<camera_name>` - 相机图像
- `observation.env_state` - 环境状态（可选）

### 动作（Action）特征

- `action` - 动作向量（如关节目标位置、末端执行器位置等）

### 其他特征

- `reward` - 奖励值（强化学习）
- `done` - 回合结束标志
- `task` - 任务名称（多任务学习）

---

## 查看状态特征的维度

### 快速查看状态维度

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/pusht")

# 查看 observation.state 的维度
if "observation.state" in ds_meta.features:
    state_feature = ds_meta.features["observation.state"]
    shape = state_feature["shape"]
    
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        dim = shape[0]
        print(f"状态特征维度: {dim}")
    else:
        print(f"状态特征形状: {shape}")
```

### 常见数据集的状态维度

不同数据集的状态维度可能不同：

- **PushT**: 通常为 2 维（x, y 位置）
- **ALOHA**: 通常为 14 维（7个关节位置 + 7个关节速度，或 8维：7个关节 + 1个夹爪）
- **LIBERO**: 8 维（7个关节 + 1个夹爪）
- **DROID**: 8 维（7个关节 + 1个夹爪）

### 使用检查脚本

已创建专门的脚本 `check_state_dim.py` 来查看状态维度：

```bash
python examples/practice/check_state_dim.py lerobot/pusht
```

输出示例：
```
============================================================
数据集: lerobot/pusht
============================================================

【方法 1: 从元数据查看】
状态特征:

  observation.state:
    形状: (2,)
    数据类型: float32
    维度数: 2

【方法 2: 从实际数据查看】
状态特征:

  observation.state:
    形状: torch.Size([2])
    数据类型: torch.float32
    维度数: 2
    值: [0.5, 0.3]

总结
============================================================
主要状态特征 'observation.state' 的维度: 2
```

## 快速检查脚本

### 检查所有特征

创建一个简单的检查脚本：

```python
#!/usr/bin/env python3
"""快速检查数据集特征"""

import sys
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from pprint import pprint

if len(sys.argv) < 2:
    print("用法: python check_features.py <dataset_repo_id>")
    print("示例: python check_features.py lerobot/pusht")
    sys.exit(1)

repo_id = sys.argv[1]

try:
    ds_meta = LeRobotDatasetMetadata(repo_id)
    
    print(f"\n数据集: {repo_id}")
    print(f"特征列表:\n")
    
    for key, feature in ds_meta.features.items():
        print(f"  {key}:")
        print(f"    形状: {feature.get('shape', 'N/A')}")
        print(f"    类型: {feature.get('dtype', 'N/A')}")
        if 'names' in feature:
            print(f"    名称: {feature['names']}")
        print()
        
except Exception as e:
    print(f"错误: {e}")
    sys.exit(1)
```

使用方法：
```bash
python check_features.py lerobot/pusht
```

### 检查状态维度

使用专门的脚本检查状态维度：

```bash
python examples/practice/check_state_dim.py lerobot/pusht
```

---

## 查看视觉特征的维度

### 视觉特征的维度结构

视觉特征（图像）通常有 **3 个维度**：

1. **高度 (Height, H)**: 图像的高度（像素数）
2. **宽度 (Width, W)**: 图像的宽度（像素数）
3. **通道数 (Channels, C)**: 颜色通道数
   - **3 通道**: RGB 彩色图像
   - **1 通道**: 灰度图像

### 维度格式说明

#### 数据集存储格式：`(H, W, C)`

在数据集中，图像通常以 `(高度, 宽度, 通道)` 格式存储：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/pusht")

# 查看视觉特征
for key, feature in ds_meta.features.items():
    if feature.get("dtype") in ["image", "video"]:
        shape = feature["shape"]  # 例如: (480, 640, 3)
        h, w, c = shape
        print(f"{key}: {h}x{w}x{c} (H×W×C)")
```

#### 策略使用格式：`(C, H, W)`

在策略中，图像转换为 PyTorch 的 channel-first 格式 `(通道, 高度, 宽度)`：

```python
from lerobot.datasets.utils import dataset_to_policy_features

policy_features = dataset_to_policy_features(ds_meta.features)

for key, feature in policy_features.items():
    if feature.type == FeatureType.VISUAL:
        shape = feature.shape  # 例如: (3, 480, 640)
        c, h, w = shape
        print(f"{key}: {c}x{h}x{w} (C×H×W)")
```

### 快速查看视觉维度

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/pusht")

# 查找所有视觉特征
for key, feature in ds_meta.features.items():
    if feature.get("dtype") in ["image", "video"]:
        shape = feature["shape"]
        if isinstance(shape, (list, tuple)) and len(shape) == 3:
            h, w, c = shape
            print(f"{key}:")
            print(f"  高度: {h} 像素")
            print(f"  宽度: {w} 像素")
            print(f"  通道数: {c}")
            print(f"  总像素数: {h * w * c}")
```

### 常见数据集的视觉特征维度

| 数据集 | 相机名称 | 分辨率 | 通道数 | 格式 |
|--------|---------|--------|--------|------|
| **PushT** | `observation.images.image` | 96×96 | 3 (RGB) | (96, 96, 3) |
| **ALOHA** | `observation.images.top` | 480×640 | 3 (RGB) | (480, 640, 3) |
| **ALOHA** | `observation.images.wrist` | 480×640 | 3 (RGB) | (480, 640, 3) |
| **LIBERO** | `observation.images.image` | 256×256 | 3 (RGB) | (256, 256, 3) |
| **LIBERO** | `observation.images.image2` | 256×256 | 3 (RGB) | (256, 256, 3) |

### 使用检查脚本

已创建专门的脚本 `check_visual_dim.py` 来查看视觉特征维度：

```bash
python examples/practice/check_visual_dim.py lerobot/pusht
```

输出示例：
```
============================================================
数据集: lerobot/pusht
============================================================

【方法 1: 从元数据查看】
视觉特征:

  observation.images.image:
    数据类型: video
    形状: (96, 96, 3)
    高度 (Height): 96
    宽度 (Width): 96
    通道数 (Channels): 3
    总像素数: 27648
    格式: (H, W, C) - 数据集存储格式

【方法 3: 从策略特征查看】
视觉特征（策略格式 - channel-first）:

  observation.images.image:
    类型: VISUAL
    形状: (3, 96, 96)
    格式: (C, H, W) - PyTorch channel-first 格式
    通道数 (Channels): 3
    高度 (Height): 96
    宽度 (Width): 96
    总像素数: 27648
```

### 维度转换说明

**数据集格式 → 策略格式**：

```python
# 数据集格式: (H, W, C) = (480, 640, 3)
# 策略格式: (C, H, W) = (3, 480, 640)

# 转换代码（在 dataset_to_policy_features 中自动完成）
h, w, c = (480, 640, 3)
policy_shape = (c, h, w)  # (3, 480, 640)
```

**实际数据中的格式**：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/pusht", episodes=[0])
sample = dataset[0]

# 查看图像形状
image_key = "observation.images.image"
if image_key in sample:
    image = sample[image_key]
    print(f"图像形状: {image.shape}")  # 可能是 (3, 96, 96) 或 (96, 96, 3)
    
    # 判断格式
    if len(image.shape) == 3:
        if image.shape[0] == 3 or image.shape[0] == 1:
            # (C, H, W) 格式
            c, h, w = image.shape
            print(f"格式: (C, H, W) = ({c}, {h}, {w})")
        elif image.shape[2] == 3 or image.shape[2] == 1:
            # (H, W, C) 格式
            h, w, c = image.shape
            print(f"格式: (H, W, C) = ({h}, {w}, {c})")
```

---

## 总结

查看强化学习特征的方法：

1. **数据集特征**：`dataset.meta.features` 或 `LeRobotDatasetMetadata(repo_id).features`
2. **策略输入特征**：`policy.config.input_features`
3. **策略输出特征**：`policy.config.output_features`
4. **特征类型**：使用 `FeatureType` 枚举进行分类

## 查看动作特征的维度

### 动作特征的维度结构

动作特征通常是 **1 维向量**，维度数取决于动作空间的大小。

### 快速查看动作维度

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/pusht")

# 查看 action 的维度
if "action" in ds_meta.features:
    action_feature = ds_meta.features["action"]
    shape = action_feature["shape"]
    names = action_feature.get("names", None)
    
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        dim = shape[0]
        print(f"动作特征维度: {dim}")
        
        # 如果有名称，显示动作组成
        if names and isinstance(names, dict) and "axes" in names:
            print("动作组成:")
            for i, name in enumerate(names['axes']):
                print(f"  [{i}] {name}")
```

### 常见数据集的动作特征维度

| 数据集 | 动作维度 | 动作组成 | 说明 |
|--------|---------|---------|------|
| **PushT** | 2 维 | `[x, y]` | 末端执行器的 x, y 位置 |
| **ALOHA** | 7 或 14 维 | `[x, y, z, roll, pitch, yaw, gripper]`<br>或 `[joint_0, ..., joint_6, gripper]` | 末端执行器位姿 + 夹爪<br>或 7个关节 + 夹爪 |
| **LIBERO** | 7 维 | `[x, y, z, roll, pitch, yaw, gripper]` | 末端执行器位姿 + 夹爪 |
| **DROID** | 8 维 | `[joint_0, ..., joint_6, gripper]` | 7个关节位置 + 夹爪位置 |

### 动作特征的类型

动作特征可以表示不同的控制方式：

1. **关节空间控制** (Joint Space)
   - 直接控制关节位置/速度
   - 例如: `[joint_0, joint_1, ..., joint_6, gripper]`
   - 维度: 关节数 + 1（夹爪）

2. **任务空间控制** (Task Space / Cartesian Space)
   - 控制末端执行器位姿
   - 例如: `[x, y, z, roll, pitch, yaw, gripper]`
   - 维度: 6（位置+姿态） + 1（夹爪） = 7

3. **混合控制**
   - 可能包含位置、速度、加速度等
   - 维度可能更大（如 14 维：7个关节位置 + 7个关节速度）

### 使用检查脚本

已创建专门的脚本 `check_action_dim.py` 来查看动作特征维度：

```bash
python examples/practice/check_action_dim.py lerobot/pusht
```

输出示例：
```
============================================================
数据集: lerobot/pusht
============================================================

【方法 1: 从元数据查看】
动作特征:

  action:
    形状: (2,)
    数据类型: float32
    维度数: 2

【方法 2: 从实际数据查看】
动作特征:

  action:
    形状: torch.Size([2])
    数据类型: torch.float32
    维度数: 2
    动作值: [0.1, 0.2]

总结
============================================================
主要动作特征 'action' 的维度: 2
```

### 查看动作特征的详细组成

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/aloha_mobile_cabinet")

if "action" in ds_meta.features:
    action_feature = ds_meta.features["action"]
    shape = action_feature["shape"]
    names = action_feature.get("names", None)
    
    print(f"动作维度: {shape[0] if isinstance(shape, (list, tuple)) else shape}")
    
    if names and isinstance(names, dict) and "axes" in names:
        print("动作组成:")
        for i, name in enumerate(names['axes']):
            print(f"  [{i}] {name}")
```

### 动作特征的时间序列

某些策略（如 Diffusion Policy）可能需要多个时间步的动作：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 使用 delta_timestamps 加载多个时间步的动作
delta_timestamps = {
    "action": [0.0, 0.1, 0.2, 0.3, 0.4]  # 当前 + 未来4步
}

dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
sample = dataset[0]

if "action" in sample:
    action = sample["action"]
    print(f"动作形状: {action.shape}")  # 例如: (5, 2)
    print(f"时间步数: {action.shape[0]}")
    print(f"动作维度: {action.shape[1]}")
```

---

## 特征维度总结

- **状态特征**: 1 维向量，维度数取决于机器人关节数（如 2, 8, 14 等）
- **视觉特征**: 3 维张量，格式为 `(H, W, C)` 或 `(C, H, W)`
  - 高度 (H): 图像高度（像素）
  - 宽度 (W): 图像宽度（像素）
  - 通道数 (C): 通常为 3 (RGB) 或 1 (灰度)
- **动作特征**: 1 维向量，维度数取决于动作空间（如 2, 7, 8 等）
  - **关节空间**: 关节数 + 夹爪（如 8 维：7个关节 + 1个夹爪）
  - **任务空间**: 6维位姿 + 夹爪（如 7 维：x, y, z, roll, pitch, yaw, gripper）
  - **混合控制**: 可能包含位置和速度（如 14 维：7个关节位置 + 7个关节速度）

## 查看奖励特征的维度

### 奖励特征的维度结构

奖励特征通常是 **标量值**（1维，shape=(1,)），数据类型为 `float32`。

### 快速查看奖励特征

```python
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("lerobot/pusht")

# 查看 reward 特征
if "reward" in ds_meta.features:
    reward_feature = ds_meta.features["reward"]
    shape = reward_feature["shape"]
    dtype = reward_feature["dtype"]
    
    print(f"奖励特征形状: {shape}")
    print(f"奖励特征数据类型: {dtype}")
    print(f"类型: 标量奖励值（每个时间步一个奖励值）")
```

### 奖励特征的特点

1. **维度**: 通常是 `(1,)` - 标量值
2. **数据类型**: `float32`
3. **键名**: 通常是 `"reward"` 或 `"next.reward"`
4. **用途**: 用于强化学习训练，评估动作的好坏

### 奖励特征的存储格式

在数据集中，奖励特征通常以以下格式存储：

```python
{
    "reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None
    }
}
```

### 奖励值的含义

- **正值**: 表示好的动作，鼓励策略采取类似动作
- **负值**: 表示不好的动作，惩罚策略采取类似动作
- **零值**: 中性动作，不奖励也不惩罚

### 注意事项

⚠️ **不是所有数据集都包含奖励特征**

- **模仿学习数据集**: 通常不包含奖励信息，因为数据来自专家演示
- **强化学习数据集**: 通常包含奖励信息，用于训练策略

### 使用检查脚本

已创建专门的脚本 `check_reward_dim.py` 来查看奖励特征：

```bash
python examples/practice/check_reward_dim.py lerobot/pusht
```

输出示例：
```
============================================================
数据集: lerobot/pusht
============================================================

【方法 1: 从元数据查看】
奖励特征:

  reward:
    形状: (1,)
    数据类型: float32
    维度数: 1
    类型: 标量奖励值

【方法 2: 从实际数据查看】
奖励特征:

  reward:
    形状: torch.Size([1])
    数据类型: torch.float32
    维度数: 1
    类型: 标量奖励值（包装在数组中）
    值: 0.0

总结
============================================================
主要奖励特征 'reward':
  形状: (1,)
  数据类型: float32
  类型: 标量奖励值（每个时间步一个奖励值）
  用途: 用于强化学习训练，评估动作的好坏
```

### 查看奖励值的统计信息

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

dataset = LeRobotDataset("lerobot/pusht", episodes=[0])

# 收集所有奖励值
rewards = []
for i in range(len(dataset)):
    sample = dataset[i]
    if "reward" in sample:
        reward = sample["reward"]
        reward_val = reward.item() if hasattr(reward, 'item') else reward
        rewards.append(reward_val)

if rewards:
    rewards = np.array(rewards)
    print(f"奖励值统计:")
    print(f"  最小值: {rewards.min()}")
    print(f"  最大值: {rewards.max()}")
    print(f"  平均值: {rewards.mean()}")
    print(f"  标准差: {rewards.std()}")
    print(f"  总和: {rewards.sum()}")
```

### 奖励特征在强化学习中的作用

1. **训练信号**: 告诉策略哪些动作是好的，哪些是坏的
2. **价值估计**: 用于估计状态或动作的价值
3. **策略优化**: 用于更新策略参数，使策略更倾向于采取高奖励的动作

### 相关特征

除了 `reward`，强化学习数据集通常还包含：

- **`done`**: 回合结束标志（bool）
- **`next.reward`**: 下一个状态的奖励（某些格式）
- **`discount`**: 折扣因子（某些格式）

---

这些信息对于理解数据格式、调试训练问题、配置策略都非常重要！

