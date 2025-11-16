# cfg.device vs 机器人主从设备（Leader/Follower）

本文档澄清 `cfg.device` 和机器人主从设备（Leader/Follower）的区别，避免混淆。

## 📋 目录

1. [核心区别](#核心区别)
2. [cfg.device - 计算设备](#cfgdevice---计算设备)
3. [机器人主从设备](#机器人主从设备)
4. [完整配置示例](#完整配置示例)
5. [常见混淆](#常见混淆)

---

## 核心区别

### 两个不同的概念

| 概念 | 含义 | 类型 | 示例 |
|------|------|------|------|
| **cfg.device** | PyTorch 计算设备 | `str` | `"cpu"`, `"cuda:0"`, `"mps"` |
| **Leader（主设备）** | 人类操作的主动臂 | `Teleoperator` | `SO100Leader` |
| **Follower（从设备）** | 被控制的从动臂 | `Robot` | `SO100Follower` |

### 关键点

- **`cfg.device`** 是**计算设备**（CPU/GPU），用于模型推理和数据处理
- **Leader/Follower** 是**机器人硬件**（主动臂/从动臂），用于物理控制

**它们是完全不同的概念！**

---

## cfg.device - 计算设备

### 定义

`cfg.device` 是 **PyTorch 的计算设备标识符**，用于指定模型和数据的计算位置。

### 类型

```python
device: str = "cpu"  # 默认值
```

### 可能的值

```python
cfg.device = "cpu"      # CPU 计算
cfg.device = "cuda:0"   # 第一个 NVIDIA GPU
cfg.device = "cuda:1"   # 第二个 NVIDIA GPU
cfg.device = "mps"      # Apple Silicon GPU (M1/M2/M3)
cfg.device = "xpu"      # Intel GPU
```

### 用途

```python
# 在处理器中使用
DeviceProcessorStep(device=cfg.device)
#    ↑ 将张量移动到指定设备（CPU/GPU）

RewardClassifierProcessorStep(device=cfg.device)
#    ↑ 奖励分类器在指定设备上运行
```

### 作用范围

- **模型推理**：策略模型在哪个设备上运行
- **数据处理**：张量存储在哪个设备上
- **加速计算**：利用 GPU 加速

---

## 机器人主从设备

### 主从控制（Leader-Follower）概念

在机器人系统中，**主从控制**是一种控制模式：

- **Leader（主设备/主动臂）**：人类操作的设备，用于输入控制信号
- **Follower（从设备/从动臂）**：被控制的机器人，跟随主设备的动作

### Leader（主设备/主动臂）

**类型**：`Teleoperator` 对象

**示例**：
- `SO100Leader` - SO-100 主动臂
- `SO101Leader` - SO-101 主动臂
- `KochLeader` - Koch 主动臂

**在代码中**：
```python
# src/lerobot/rl/gym_manipulator.py (334)
teleop_device = make_teleoperator_from_config(cfg.teleop)
#    ↑ 如果 cfg.teleop.type = "so100_leader"
#    teleop_device = SO100Leader(...)
```

### Follower（从设备/从动臂）

**类型**：`Robot` 对象

**示例**：
- `SO100Follower` - SO-100 从动臂
- `SO101Follower` - SO-101 从动臂
- `KochFollower` - Koch 从动臂

**在代码中**：
```python
# src/lerobot/rl/gym_manipulator.py (333)
robot = make_robot_from_config(cfg.robot)
#    ↑ 如果 cfg.robot.type = "so100_follower"
#    robot = SO100Follower(...)
```

### 主从控制流程

```
人类操作 Leader（主动臂）
    ↓
Leader 读取人类动作
    ↓
teleop_device.get_action()
    ↓
动作传递给 Follower（从动臂）
    ↓
robot.send_action(action)
    ↓
Follower 执行动作
```

---

## 完整配置示例

### 示例：SO-100 主从控制配置

```python
# 配置示例
cfg = GymManipulatorConfig(
    env=HILSerlRobotEnvConfig(
        # 从设备（Follower）- 被控制的机器人
        robot=RobotConfig(
            type="so100_follower",  # ← 从设备
            port="/dev/tty.usbmodem5A460814411",
            id="follower_arm"
        ),
        
        # 主设备（Leader）- 人类操作的主动臂
        teleop=TeleoperatorConfig(
            type="so100_leader",    # ← 主设备
            port="/dev/tty.usbmodem5A460819811",
            id="leader_arm"
        ),
    ),
    
    # 计算设备（与主从设备无关）
    device="cuda:0",  # ← PyTorch 计算设备
)

# 创建环境
env, teleop_device = make_robot_env(cfg.env)
#    ↑ env.robot = SO100Follower(...)  ← 从设备
#    ↑ teleop_device = SO100Leader(...)  ← 主设备

# 创建处理器
env_processor, action_processor = make_processors(
    env,
    teleop_device,  # ← 主设备（Leader）
    cfg.env,
    cfg.device      # ← 计算设备（"cuda:0"）
)
```

### 代码流程

```python
# 1. 创建从设备（Follower）
robot = make_robot_from_config(cfg.robot)
#    → SO100Follower(...)  ← 从设备

# 2. 创建主设备（Leader）
teleop_device = make_teleoperator_from_config(cfg.teleop)
#    → SO100Leader(...)  ← 主设备

# 3. 创建环境（包含从设备）
env = RobotEnv(robot=robot)
#    ↑ env.robot = SO100Follower（从设备）

# 4. 创建处理器（使用主设备和计算设备）
env_processor, action_processor = make_processors(
    env,              # ← 包含从设备
    teleop_device,    # ← 主设备
    cfg.env,
    cfg.device        # ← 计算设备（"cuda:0"）
)
```

---

## 常见混淆

### ❌ 错误理解

**错误**：`cfg.device` 应该是机器人的从设备（follower）

**原因**：混淆了计算设备和机器人硬件设备

### ✅ 正确理解

**正确**：
- **`cfg.device`** = PyTorch 计算设备（`"cpu"`, `"cuda:0"` 等）
- **`robot`** = 机器人的从设备（`SO100Follower` 等）
- **`teleop_device`** = 机器人的主设备（`SO100Leader` 等）

### 三个不同的"设备"概念

```
1. cfg.device (计算设备)
   └── PyTorch 设备：CPU/GPU
   └── 用于：模型推理、数据处理
   └── 示例："cuda:0"

2. teleop_device (主设备/Leader)
   └── 机器人硬件：主动臂
   └── 用于：人类操作、输入控制
   └── 示例：SO100Leader

3. robot (从设备/Follower)
   └── 机器人硬件：从动臂
   └── 用于：执行动作、被控制
   └── 示例：SO100Follower
```

---

## 代码中的对应关系

### 配置到对象的映射

```python
# 配置
cfg = GymManipulatorConfig(
    env=HILSerlRobotEnvConfig(
        robot=RobotConfig(type="so100_follower"),      # ← 从设备配置
        teleop=TeleoperatorConfig(type="so100_leader"), # ← 主设备配置
    ),
    device="cuda:0",  # ← 计算设备配置
)

# 创建对象
robot = make_robot_from_config(cfg.robot)
#    → SO100Follower(...)  ← 从设备对象

teleop_device = make_teleoperator_from_config(cfg.teleop)
#    → SO100Leader(...)  ← 主设备对象

# 使用计算设备
DeviceProcessorStep(device=cfg.device)
#    → device="cuda:0"  ← 计算设备字符串
```

### 在 make_processors 中的使用

```python
# src/lerobot/rl/gym_manipulator.py (755)
env_processor, action_processor = make_processors(
    env,              # ← 包含 robot (从设备)
    teleop_device,    # ← 主设备 (Leader)
    cfg.env,
    cfg.device        # ← 计算设备 ("cuda:0")
)
```

---

## 总结

### cfg.device（计算设备）

- **类型**：`str` 字符串
- **用途**：PyTorch 计算设备
- **示例**：`"cpu"`, `"cuda:0"`, `"mps"`
- **作用**：指定模型和数据的计算位置

### teleop_device（主设备/Leader）

- **类型**：`Teleoperator` 对象
- **用途**：人类操作的主动臂
- **示例**：`SO100Leader`, `SO101Leader`
- **作用**：获取人类动作，检测干预

### robot（从设备/Follower）

- **类型**：`Robot` 对象
- **用途**：被控制的从动臂
- **示例**：`SO100Follower`, `SO101Follower`
- **作用**：执行动作，跟随主设备

### 关键区别

1. **`cfg.device`** 是**计算设备**（CPU/GPU），与机器人硬件无关
2. **`teleop_device`** 是**主设备**（Leader），人类操作的主动臂
3. **`robot`** 是**从设备**（Follower），被控制的从动臂

**它们是完全不同的概念！**

---

## 命名说明

### 为什么容易混淆？

- **"device"** 这个词在英语中既可以指硬件设备，也可以指计算设备
- 在机器人领域，"device" 通常指硬件设备
- 在深度学习领域，"device" 通常指计算设备（CPU/GPU）

### 在 LeRobot 中的命名

- **`cfg.device`**：遵循 PyTorch 的命名习惯，指计算设备
- **`teleop_device`**：指遥操作硬件设备（主设备）
- **`robot`**：指机器人硬件设备（从设备）

### 建议

为了避免混淆，可以这样理解：
- **`cfg.device`** = **计算设备**（Computing Device）
- **`teleop_device`** = **主设备**（Leader Device）
- **`robot`** = **从设备**（Follower Device）

---

这些概念虽然都叫"设备"，但含义完全不同！

