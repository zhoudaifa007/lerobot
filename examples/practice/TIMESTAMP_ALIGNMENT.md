# LeRobot 数据时间对齐机制

本文档详细说明不同类型的数据在写入和读取 `LeRobotDataset` 时如何实现时间对齐。

## 📋 目录

1. [时间对齐概述](#时间对齐概述)
2. [写入时的时间对齐](#写入时的时间对齐)
3. [读取时的时间对齐](#读取时的时间对齐)
4. [FPS 同步机制](#fps-同步机制)
5. [视频帧的时间对齐](#视频帧的时间对齐)
6. [代码示例](#代码示例)

---

## 时间对齐概述

LeRobot 使用 **统一的时间戳系统** 来对齐不同类型的数据：

- **状态数据** (observation.state)
- **视觉数据** (observation.images.*)
- **动作数据** (action)
- **奖励数据** (reward)
- **其他元数据**

所有数据在同一个时间步（frame）使用相同的 `timestamp` 进行标记。

---

## 写入时的时间对齐

### 核心机制

在 `add_frame()` 方法中，所有不同类型的数据都在**同一个时间步**写入：

```python
def add_frame(self, frame: dict) -> None:
    # 1. 自动添加 frame_index 和 timestamp
    frame_index = self.episode_buffer["size"]
    timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
    
    # 2. 所有数据使用相同的 timestamp
    self.episode_buffer["frame_index"].append(frame_index)
    self.episode_buffer["timestamp"].append(timestamp)
    
    # 3. 将所有特征添加到同一个帧
    for key in frame:
        if self.features[key]["dtype"] in ["image", "video"]:
            # 图像数据：保存到临时目录，稍后编码为视频
            self._save_image(frame[key], img_path)
            self.episode_buffer[key].append(str(img_path))
        else:
            # 其他数据：直接添加到缓冲区
            self.episode_buffer[key].append(frame[key])
```

### 时间戳生成规则

1. **如果提供了 timestamp**：使用提供的值
2. **如果没有提供 timestamp**：自动计算 `timestamp = frame_index / fps`

### 数据记录流程

在 `lerobot_record.py` 的记录循环中：

```python
def record_loop(...):
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        
        # 1. 获取观察（状态 + 图像）
        obs = robot.get_observation()
        observation_frame = build_dataset_frame(
            dataset.features, obs_processed, prefix=OBS_STR
        )
        
        # 2. 获取动作
        action_values = act_processed_teleop  # 或 act_processed_policy
        action_frame = build_dataset_frame(
            dataset.features, action_values, prefix=ACTION
        )
        
        # 3. 组合所有数据到同一帧
        frame = {
            **observation_frame,  # 包含 observation.state 和 observation.images.*
            **action_frame,        # 包含 action
            "task": single_task
        }
        
        # 4. 添加时间戳（可选，如果不提供会自动计算）
        # frame["timestamp"] = time.perf_counter() - start_episode_t
        
        # 5. 写入数据集（所有数据使用相同的时间戳）
        dataset.add_frame(frame)
        
        # 6. 控制循环频率
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        
        timestamp = time.perf_counter() - start_episode_t
```

### 关键点

1. **同步采样**：所有数据在同一个循环迭代中采集
2. **统一时间戳**：同一帧的所有数据共享相同的 `timestamp`
3. **FPS 控制**：通过 `busy_wait()` 确保采样频率为 `fps`

---

## 读取时的时间对齐

### 基本读取

默认情况下，每个样本包含同一时间步的所有数据：

```python
dataset = LeRobotDataset("lerobot/pusht")
sample = dataset[0]

# 所有数据都来自同一时间步
print(sample["observation.state"])    # 状态
print(sample["observation.images.image"])  # 图像
print(sample["action"])               # 动作
print(sample["timestamp"])            # 时间戳
```

### 使用 delta_timestamps 查询历史/未来帧

某些策略（如 Diffusion Policy）需要多个时间步的数据：

```python
delta_timestamps = {
    "observation.state": [-1.0, -0.5, 0.0],  # 1秒前、0.5秒前、当前
    "action": [0.0, 0.1, 0.2, 0.3],          # 当前、未来0.1s、0.2s、0.3s
    "observation.images.image": [-0.2, 0.0]  # 0.2秒前、当前
}

dataset = LeRobotDataset(
    "lerobot/pusht",
    delta_timestamps=delta_timestamps
)

sample = dataset[0]

# observation.state 现在是 (3, state_dim) - 3个时间步
print(sample["observation.state"].shape)

# action 现在是 (4, action_dim) - 4个时间步
print(sample["action"].shape)

# observation.images.image 现在是 (2, C, H, W) - 2个时间步
print(sample["observation.images.image"].shape)
```

### 时间对齐算法

在 `OnlineBuffer` 和 `StreamingLeRobotDataset` 中，使用以下算法对齐时间：

```python
def _align_timestamps(current_ts, delta_timestamps, episode_timestamps, tolerance_s):
    """
    对齐时间戳的核心算法
    
    Args:
        current_ts: 当前帧的时间戳
        delta_timestamps: 需要查询的时间偏移列表
        episode_timestamps: 该回合所有帧的时间戳
        tolerance_s: 允许的时间误差
    """
    # 1. 计算查询时间戳
    query_ts = current_ts + delta_timestamps
    
    # 2. 计算距离矩阵
    dist = np.abs(query_ts[:, None] - episode_timestamps[None, :])
    
    # 3. 找到最接近的帧索引
    argmin_ = np.argmin(dist, axis=1)
    min_ = dist[np.arange(dist.shape[0]), argmin_]
    
    # 4. 检查是否在容差范围内
    is_pad = min_ > tolerance_s
    
    # 5. 验证超出容差的查询是否在回合范围外
    assert (
        (query_ts[is_pad] < episode_timestamps[0]) | 
        (episode_timestamps[-1] < query_ts[is_pad])
    ).all(), "时间戳超出容差范围"
    
    return argmin_, is_pad
```

---

## FPS 同步机制

### FPS 的作用

FPS (Frames Per Second) 定义了数据采样的频率：

- **FPS = 30**：每秒采样 30 次，时间间隔 = 1/30 ≈ 0.033 秒
- **FPS = 10**：每秒采样 10 次，时间间隔 = 0.1 秒

### 时间戳验证

在数据集初始化时，会验证时间戳是否符合 FPS：

```python
def __init__(self, ..., tolerance_s: float = 1e-4):
    """
    tolerance_s: 用于确保时间戳与 fps 值同步的容差（秒）
    
    用于检查：
    1. 每个时间戳与下一个时间戳的间隔是否为 1/fps +/- tolerance_s
    2. delta_timestamps 是否为 1/fps 的倍数
    """
    self.tolerance_s = tolerance_s
    # 验证时间戳间隔
    self._validate_timestamps()
```

### 时间戳间隔检查

```python
def _validate_timestamps(self):
    """验证时间戳间隔是否符合 FPS"""
    timestamps = self.hf_dataset["timestamp"]
    
    for i in range(len(timestamps) - 1):
        dt = timestamps[i + 1] - timestamps[i]
        expected_dt = 1.0 / self.fps
        
        if abs(dt - expected_dt) > self.tolerance_s:
            raise ValueError(
                f"时间戳间隔不符合 FPS: "
                f"期望 {expected_dt:.6f}s, 实际 {dt:.6f}s, "
                f"误差 {abs(dt - expected_dt):.6f}s > {self.tolerance_s}"
            )
```

---

## 视频帧的时间对齐

### 视频编码

在写入时，图像帧被保存到临时目录，然后编码为视频：

```python
def save_episode(self):
    # 1. 保存所有非视频数据到 parquet
    # 2. 将临时图像目录编码为视频
    for video_key in self.meta.video_keys:
        video_path = self._encode_temporary_episode_video(
            video_key, episode_index
        )
```

### 视频解码

在读取时，根据时间戳从视频中提取帧：

```python
def decode_video_frames(video_path, timestamps, tolerance_s):
    """
    从视频中提取指定时间戳的帧
    
    Args:
        video_path: 视频文件路径
        timestamps: 要提取的时间戳列表
        tolerance_s: 允许的时间误差
    """
    # 1. 加载视频帧（从关键帧开始）
    decoder = get_decoder(video_path)
    frame_indices = [round(ts * fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)
    
    # 2. 获取实际加载的时间戳
    loaded_ts = [frame.pts_seconds for frame in frames_batch]
    
    # 3. 计算查询时间戳和实际时间戳的距离
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)
    
    # 4. 验证是否在容差范围内
    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), "视频帧时间戳超出容差"
    
    # 5. 返回最接近的帧
    return frames_batch[argmin_]
```

### 关键点

1. **关键帧机制**：视频使用关键帧压缩，需要从关键帧开始解码
2. **时间戳匹配**：使用最近邻搜索找到最接近的帧
3. **容差检查**：确保提取的帧在时间容差范围内

---

## 代码示例

### 示例 1：记录数据（写入时对齐）

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR

# 创建数据集
dataset = LeRobotDataset.create(
    repo_id="my_dataset",
    fps=30,  # 30 FPS
    features={
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "observation.images.camera": {"dtype": "video", "shape": (480, 640, 3)},
        "action": {"dtype": "float32", "shape": (7,)},
    }
)

# 记录循环
for frame_idx in range(100):
    # 1. 获取观察（状态 + 图像）
    state = robot.get_joint_positions()  # (8,)
    image = camera.get_image()          # (480, 640, 3)
    
    observation_frame = build_dataset_frame(
        dataset.features,
        {"state": state, "images.camera": image},
        prefix=OBS_STR
    )
    
    # 2. 获取动作
    action = get_action()  # (7,)
    action_frame = build_dataset_frame(
        dataset.features,
        action,
        prefix=ACTION
    )
    
    # 3. 组合所有数据（所有数据使用相同的时间戳）
    frame = {
        **observation_frame,
        **action_frame,
        "task": "pick_and_place"
        # timestamp 会自动计算为 frame_idx / fps
    }
    
    # 4. 写入数据集
    dataset.add_frame(frame)

# 5. 保存回合
dataset.save_episode()
```

### 示例 2：读取数据（读取时对齐）

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 基本读取（同一时间步的所有数据）
dataset = LeRobotDataset("lerobot/pusht")
sample = dataset[0]

print(f"时间戳: {sample['timestamp']}")
print(f"状态: {sample['observation.state'].shape}")  # (8,)
print(f"图像: {sample['observation.images.image'].shape}")  # (3, 96, 96)
print(f"动作: {sample['action'].shape}")  # (2,)

# 使用 delta_timestamps 查询历史/未来帧
delta_timestamps = {
    "observation.state": [-1.0, -0.5, 0.0],  # 历史帧
    "action": [0.0, 0.1, 0.2, 0.3],          # 未来帧
}

dataset_with_history = LeRobotDataset(
    "lerobot/pusht",
    delta_timestamps=delta_timestamps,
    tolerance_s=1e-4
)

sample = dataset_with_history[0]

# 现在数据包含多个时间步
print(f"状态（3个时间步）: {sample['observation.state'].shape}")  # (3, 8)
print(f"动作（4个时间步）: {sample['action'].shape}")  # (4, 2)
```

### 示例 3：验证时间对齐

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

dataset = LeRobotDataset("lerobot/pusht")

# 检查时间戳间隔
timestamps = [dataset[i]["timestamp"].item() for i in range(100)]
dts = np.diff(timestamps)

fps = dataset.fps
expected_dt = 1.0 / fps

print(f"FPS: {fps}")
print(f"期望时间间隔: {expected_dt:.6f}s")
print(f"实际时间间隔: 平均 {np.mean(dts):.6f}s, 标准差 {np.std(dts):.6f}s")
print(f"最大误差: {np.max(np.abs(dts - expected_dt)):.6f}s")

# 验证所有数据在同一时间步
for i in range(10):
    sample = dataset[i]
    ts = sample["timestamp"].item()
    print(f"样本 {i}: timestamp={ts:.6f}")
    # 所有特征都应该有相同的 timestamp（在同一个帧中）
```

---

## 总结

### 写入时的时间对齐

1. **统一时间戳**：所有数据在 `add_frame()` 中使用相同的 `timestamp`
2. **自动计算**：如果没有提供 timestamp，自动计算为 `frame_index / fps`
3. **同步采样**：在同一个循环迭代中采集所有数据

### 读取时的时间对齐

1. **默认对齐**：同一索引的所有数据来自同一时间步
2. **delta_timestamps**：可以查询历史或未来的帧
3. **容差匹配**：使用 `tolerance_s` 来匹配最接近的时间戳

### 关键参数

- **fps**: 采样频率，决定时间戳间隔
- **tolerance_s**: 时间容差，用于验证和匹配时间戳（默认 1e-4 秒）
- **delta_timestamps**: 用于查询多个时间步的数据

### 最佳实践

1. **记录时**：确保所有数据在同一个循环迭代中采集
2. **读取时**：使用 `delta_timestamps` 时，确保偏移是 `1/fps` 的倍数
3. **验证**：检查时间戳间隔是否符合 FPS 要求
4. **容差**：根据实际需求调整 `tolerance_s`（默认 1e-4 秒通常足够）

---

这些机制确保了不同类型的数据在时间上完全对齐，这对于训练和推理都至关重要！

