# teleop_device 和 cfg.device 的区别

本文档详细说明 `teleop_device` 和 `cfg.device` 这两个参数的含义和用途。

## 📋 目录

1. [teleop_device - 遥操作设备](#teleop_device---遥操作设备)
2. [cfg.device - 计算设备](#cfgdevice---计算设备)
3. [两者的区别](#两者的区别)
4. [使用示例](#使用示例)

---

## teleop_device - 遥操作设备

### 定义

`teleop_device` 是一个 **Teleoperator** 对象，用于**人工操作机器人**的设备。

### 类型

```python
teleop_device: Teleoperator | None
```

### 用途

1. **人工干预（Human-in-the-Loop）**：允许人类操作者控制机器人
2. **数据录制**：用于录制专家演示数据
3. **干预检测**：检测人类是否介入控制

### 支持的遥操作设备类型

LeRobot 支持多种遥操作设备：

| 设备类型 | 说明 | 使用场景 |
|---------|------|---------|
| **so100_leader** | SO-100 主动臂 | 主从控制，精确操作 |
| **so101_leader** | SO-101 主动臂 | 主从控制，精确操作 |
| **phone** | 手机（iOS/Android） | 通过手机 AR 控制 |
| **gamepad** | 游戏手柄 | 简单操作，游戏风格控制 |
| **keyboard** | 键盘 | 开发调试，简单控制 |
| **keyboard_ee** | 键盘（末端执行器） | 直接控制末端执行器 |
| **koch_leader** | Koch 主动臂 | 主从控制 |
| **homunculus_glove** | Homunculus 手套 | 手势控制 |
| **homunculus_arm** | Homunculus 手臂 | 手臂控制 |
| **bi_so100_leader** | 双臂 SO-100 | 双臂协调控制 |
| **reachy2_teleoperator** | Reachy2 遥操作器 | Reachy2 机器人控制 |

### 创建方式

```python
# src/lerobot/rl/gym_manipulator.py (334)
teleop_device = make_teleoperator_from_config(cfg.teleop)
teleop_device.connect()
```

### 在代码中的使用

```python
# src/lerobot/rl/gym_manipulator.py (463-464)
action_pipeline_steps = [
    AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
    #    ↑ 将遥操作设备的动作添加到补充数据中
    AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
    #    ↑ 将遥操作事件（如干预、成功）添加到信息中
    InterventionActionProcessorStep(...),
    #    ↑ 处理人工干预，可能覆盖策略动作
]
```

### Teleoperator 接口

```python
# src/lerobot/teleoperators/teleoperator.py
class Teleoperator(abc.ABC):
    """遥操作设备的基类"""
    
    @property
    def action_features(self) -> dict:
        """返回动作特征（动作空间定义）"""
        pass
    
    def get_action(self) -> dict:
        """获取当前动作（从遥操作设备读取）"""
        pass
    
    def connect(self) -> None:
        """连接遥操作设备"""
        pass
    
    def disconnect(self) -> None:
        """断开连接"""
        pass
```

### 使用场景

1. **数据录制**：人类操作者使用遥操作设备控制机器人，录制演示数据
2. **人工干预训练**：在强化学习训练中，人类可以随时介入纠正策略
3. **测试和调试**：使用键盘或游戏手柄快速测试机器人

---

## cfg.device - 计算设备

### 定义

`cfg.device` 是一个**字符串**，指定用于**计算**的设备**（CPU、GPU 等）。

### 类型

```python
device: str = "cpu"  # 默认值
```

### 可能的值

| 值 | 说明 | 示例 |
|---|------|------|
| `"cpu"` | CPU 计算 | 默认值，适用于所有系统 |
| `"cuda"` | NVIDIA GPU | `"cuda"` 或 `"cuda:0"` |
| `"cuda:0"` | 第一个 GPU | 多 GPU 系统中的第一个 |
| `"cuda:1"` | 第二个 GPU | 多 GPU 系统中的第二个 |
| `"mps"` | Apple Silicon GPU | macOS M1/M2/M3 芯片 |
| `"xpu"` | Intel GPU | Intel GPU 加速 |

### 用途

1. **模型计算**：指定策略模型在哪个设备上运行
2. **数据处理**：指定张量在哪个设备上处理
3. **奖励分类器**：指定奖励分类器在哪个设备上运行

### 在代码中的使用

```python
# src/lerobot/rl/gym_manipulator.py (384, 460, 452)
env_pipeline_steps = [
    # ...
    DeviceProcessorStep(device=device),  # ← 将数据移动到指定设备
    #    ↑ device = cfg.device
]

# 奖励分类器也使用这个设备
RewardClassifierProcessorStep(
    pretrained_path=...,
    device=device,  # ← 奖励分类器在指定设备上运行
    #    ↑ device = cfg.device
)
```

### 设备选择逻辑

```python
# src/lerobot/utils/utils.py
def get_safe_torch_device(try_device: str) -> torch.device:
    """安全地获取 PyTorch 设备"""
    if try_device.startswith("cuda"):
        assert torch.cuda.is_available()
        return torch.device(try_device)
    elif try_device == "mps":
        assert torch.backends.mps.is_available()
        return torch.device("mps")
    elif try_device == "cpu":
        return torch.device("cpu")
    # ...
```

---

## 两者的区别

### 对比表

| 特性 | teleop_device | cfg.device |
|------|--------------|------------|
| **类型** | `Teleoperator` 对象 | `str` 字符串 |
| **用途** | 人工操作机器人 | 指定计算设备 |
| **物理存在** | 是（硬件设备） | 否（逻辑设备） |
| **示例值** | `SO100Leader(...)` | `"cuda:0"` |
| **作用范围** | 控制流程 | 计算流程 |
| **是否必需** | 可选（可为 None） | 必需（默认 "cpu"） |

### 功能对比

```
teleop_device (遥操作设备)
├── 用途：人工控制机器人
├── 类型：硬件设备接口
├── 功能：
│   ├── 获取人类动作
│   ├── 检测干预事件
│   └── 标记任务成功
└── 示例：SO-100 主动臂、手机、游戏手柄

cfg.device (计算设备)
├── 用途：指定计算位置
├── 类型：字符串标识符
├── 功能：
│   ├── 模型推理位置
│   ├── 张量存储位置
│   └── 加速计算
└── 示例："cpu", "cuda:0", "mps"
```

---

## 使用示例

### 示例 1：创建遥操作设备和环境

```python
# src/lerobot/rl/gym_manipulator.py (754-755)
env, teleop_device = make_robot_env(cfg.env)
#    ↑ 返回 (环境, 遥操作设备)
#    teleop_device 可能是 SO100Leader、Phone、Gamepad 等

env_processor, action_processor = make_processors(
    env, 
    teleop_device,  # ← 传入遥操作设备
    cfg.env, 
    cfg.device      # ← 传入计算设备（如 "cuda:0"）
)
```

### 示例 2：遥操作设备的使用

```python
# 在动作处理器中使用
action_pipeline_steps = [
    # 1. 将遥操作设备的动作添加到补充数据
    AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
    #    ↑ teleop_device.get_action() 获取人类动作
    
    # 2. 将遥操作事件添加到信息
    AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
    #    ↑ 检测干预、成功等事件
    
    # 3. 处理干预（如果人类介入，使用人类动作）
    InterventionActionProcessorStep(...),
    #    ↑ 从补充数据中读取 teleop_action
]
```

### 示例 3：计算设备的使用

```python
# 在环境处理器中使用
env_pipeline_steps = [
    VanillaObservationProcessorStep(),
    # ...
    
    # 将数据移动到指定设备
    DeviceProcessorStep(device=cfg.device),
    #    ↑ 如果 cfg.device = "cuda:0"，数据会移动到 GPU
    
    # 奖励分类器在指定设备上运行
    RewardClassifierProcessorStep(
        pretrained_path="...",
        device=cfg.device,  # ← 分类器在 GPU 上运行
        #    ↑ 如果 cfg.device = "cuda:0"，分类器在 GPU 上
    ),
]
```

### 示例 4：完整配置

```python
# 配置示例
cfg = GymManipulatorConfig(
    env=HILSerlRobotEnvConfig(
        robot=RobotConfig(type="so100_follower", ...),
        teleop=TeleoperatorConfig(type="so100_leader", ...),
        #    ↑ 配置遥操作设备类型
    ),
    device="cuda:0",  # ← 配置计算设备
    #    ↑ 使用第一个 GPU
)

# 创建环境
env, teleop_device = make_robot_env(cfg.env)
#    ↑ teleop_device = SO100Leader(...) 对象

# 创建处理器
env_processor, action_processor = make_processors(
    env,
    teleop_device,  # ← SO100Leader 对象
    cfg.env,
    cfg.device      # ← "cuda:0" 字符串
)
```

---

## 代码流程

### 遥操作设备的流程

```
1. make_robot_env(cfg.env)
   → teleop_device = make_teleoperator_from_config(cfg.teleop)
   → teleop_device.connect()
   
2. make_processors(..., teleop_device, ...)
   → AddTeleopActionAsComplimentaryDataStep(teleop_device)
   → AddTeleopEventsAsInfoStep(teleop_device)
   
3. control_loop(..., teleop_device, ...)
   → 在循环中：
      - teleop_device.get_action() 获取人类动作
      - 检测干预事件
      - 如果干预，使用人类动作覆盖策略动作
```

### 计算设备的流程

```
1. cfg.device = "cuda:0"  # 配置
   
2. make_processors(..., device=cfg.device)
   → DeviceProcessorStep(device="cuda:0")
   → RewardClassifierProcessorStep(device="cuda:0")
   
3. 数据处理时：
   → 张量移动到 GPU: tensor.to("cuda:0")
   → 模型在 GPU 上运行
   → 奖励分类器在 GPU 上推理
```

---

## 总结

### teleop_device（遥操作设备）

- **类型**：`Teleoperator` 对象
- **用途**：人工操作机器人的硬件设备
- **示例**：SO-100 主动臂、手机、游戏手柄、键盘
- **功能**：
  - 获取人类动作
  - 检测干预事件
  - 标记任务成功
- **是否必需**：可选（可为 `None`）

### cfg.device（计算设备）

- **类型**：`str` 字符串
- **用途**：指定计算设备（CPU/GPU）
- **示例**：`"cpu"`, `"cuda:0"`, `"mps"`
- **功能**：
  - 指定模型运行位置
  - 指定张量存储位置
  - 加速计算
- **是否必需**：必需（默认 `"cpu"`）

### 关键区别

1. **teleop_device** 是**硬件设备接口**，用于**控制**
2. **cfg.device** 是**计算设备标识**，用于**计算**

两者在代码中**同时使用**，但**作用不同**：
- `teleop_device` 用于获取人类输入和检测干预
- `cfg.device` 用于指定模型和数据的计算位置

---

这些参数使得 LeRobot 能够同时支持人工控制和高效计算！

