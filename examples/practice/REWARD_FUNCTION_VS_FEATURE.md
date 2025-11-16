# 奖励函数 vs 奖励特征

本文档详细说明奖励函数和奖励特征的区别，以及如何在 LeRobot 中定义和使用它们。

## 📋 目录

1. [核心概念](#核心概念)
2. [奖励函数（Reward Function）](#奖励函数reward-function)
3. [奖励特征（Reward Feature）](#奖励特征reward-feature)
4. [两者的关系](#两者的关系)
5. [如何定义奖励函数](#如何定义奖励函数)
6. [奖励处理器](#奖励处理器)
7. [代码示例](#代码示例)

---

## 核心概念

### 奖励函数 vs 奖励特征

| 特性 | 奖励函数 | 奖励特征 |
|------|---------|---------|
| **定义** | 计算奖励的数学函数 | 数据集中存储的奖励值 |
| **位置** | 环境（Environment）中 | 数据集（Dataset）中 |
| **时机** | 运行时计算 | 已计算并存储 |
| **输入** | `(state, action, next_state)` | 无（已经是值） |
| **输出** | 标量奖励值 | 标量奖励值 |
| **用途** | 强化学习训练/评估 | 离线训练、数据分析 |

---

## 奖励函数（Reward Function）

### 定义

**奖励函数**是在环境中定义的函数，用于根据当前状态、动作和下一个状态计算奖励值。

### 特点

1. **运行时计算**：每次环境执行 `step()` 时计算
2. **动态性**：可以根据环境状态实时计算
3. **可修改**：可以通过奖励处理器修改
4. **环境相关**：不同环境有不同的奖励函数

### 在环境中的实现

奖励函数通常在环境的 `step()` 方法中计算：

```python
class MyEnv(gym.Env):
    def step(self, action):
        # 1. 执行动作
        next_state = self._execute_action(action)
        
        # 2. 计算奖励（奖励函数）
        reward = self._compute_reward(self.state, action, next_state)
        
        # 3. 检查是否结束
        done = self._is_done(next_state)
        
        return next_state, reward, done, info
    
    def _compute_reward(self, state, action, next_state):
        """奖励函数：根据状态和动作计算奖励"""
        # 示例：距离目标的奖励
        distance_to_goal = np.linalg.norm(next_state - self.goal)
        reward = -distance_to_goal  # 距离越近，奖励越高
        
        # 示例：任务完成的奖励
        if self._is_goal_reached(next_state):
            reward += 10.0
        
        # 示例：碰撞惩罚
        if self._is_collision(next_state):
            reward -= 5.0
        
        return reward
```

### LeRobot 中的奖励函数示例

#### 1. Metaworld 环境

```python
# src/lerobot/envs/metaworld.py
class MetaworldEnv(gym.Env):
    def step(self, action):
        # 环境内部计算奖励
        raw_obs, reward, done, truncated, info = self._env.step(action)
        
        # reward 是环境返回的奖励值（由环境的奖励函数计算）
        return observation, reward, terminated, truncated, info
```

#### 2. LIBERO 环境

```python
# src/lerobot/envs/libero.py
class LiberoEnv(gym.Env):
    def step(self, action):
        # 环境内部计算奖励
        raw_obs, reward, done, info = self._env.step(action)
        
        # reward 是环境返回的奖励值
        return observation, reward, terminated, truncated, info
```

---

## 奖励特征（Reward Feature）

### 定义

**奖励特征**是数据集中存储的奖励值，是已经计算好的标量值。

### 特点

1. **静态存储**：已经计算并存储在数据集中
2. **不可修改**：数据集中的值不会改变
3. **离线使用**：用于离线训练和数据分析
4. **格式固定**：通常是 `float32`，形状为 `(1,)`

### 在数据集中的存储

```python
# 数据集特征定义
features = {
    "reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None
    }
}

# 数据集中的实际值
sample = dataset[0]
reward = sample["reward"]  # 例如: tensor([0.5])
```

### 从奖励函数到奖励特征

在记录数据时，奖励函数计算的奖励值会被存储为奖励特征：

```python
# 记录循环
while recording:
    # 1. 执行动作
    obs, reward, done, info = env.step(action)
    #    ↑ 奖励函数计算的奖励值
    
    # 2. 存储到数据集
    frame = {
        "observation.state": obs,
        "action": action,
        "reward": reward,  # ← 奖励函数的值存储为奖励特征
        "done": done
    }
    dataset.add_frame(frame)
```

---

## 两者的关系

### 流程图

```
┌─────────────────┐
│   环境 (Env)    │
│                 │
│  step(action)   │
│       ↓         │
│  奖励函数计算    │
│  reward = f(...) │
└────────┬────────┘
         │
         ↓ reward 值
┌─────────────────┐
│  记录到数据集    │
│                 │
│  dataset.add_   │
│  frame({        │
│    "reward":    │
│    reward       │
│  })             │
└────────┬────────┘
         │
         ↓ 存储
┌─────────────────┐
│   数据集存储     │
│                 │
│  reward 特征     │
│  (静态值)        │
└─────────────────┘
```

### 关键区别

1. **奖励函数**：
   - 在环境中定义
   - 运行时计算
   - 可以修改（通过处理器）
   - 用于在线强化学习

2. **奖励特征**：
   - 在数据集中存储
   - 已经计算好的值
   - 不可修改
   - 用于离线训练

---

## 如何定义奖励函数

### 方法 1：在自定义环境中定义

```python
import gym
import numpy as np

class CustomRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.goal = np.array([0.5, 0.5, 0.5])
        self.state = None
        
    def reset(self):
        self.state = np.random.rand(3)
        return self.state
    
    def step(self, action):
        # 执行动作
        self.state = self.state + action * 0.1
        
        # 定义奖励函数
        reward = self._compute_reward(self.state, action)
        
        # 检查是否完成
        done = np.linalg.norm(self.state - self.goal) < 0.1
        
        return self.state, reward, done, {}
    
    def _compute_reward(self, state, action):
        """自定义奖励函数"""
        # 1. 距离目标的奖励
        distance = np.linalg.norm(state - self.goal)
        distance_reward = -distance
        
        # 2. 任务完成的奖励
        if distance < 0.1:
            completion_reward = 10.0
        else:
            completion_reward = 0.0
        
        # 3. 动作平滑性奖励（惩罚大动作）
        action_penalty = -0.1 * np.linalg.norm(action)
        
        # 总奖励
        total_reward = distance_reward + completion_reward + action_penalty
        
        return total_reward
```

### 方法 2：使用奖励处理器修改奖励

```python
from lerobot.processor.pipeline import RewardProcessorStep

class CustomRewardProcessor(RewardProcessorStep):
    """自定义奖励处理器"""
    
    def reward(self, reward):
        """修改奖励值"""
        # 例如：将奖励缩放到 [0, 1]
        normalized_reward = (reward + 1.0) / 2.0
        return normalized_reward
```

### 方法 3：使用奖励分类器

LeRobot 提供了 `RewardClassifierProcessorStep`，可以根据图像分类器预测成功来修改奖励：

```python
# src/lerobot/processor/hil_processor.py
@dataclass
class RewardClassifierProcessorStep(ProcessorStep):
    """使用奖励分类器修改奖励"""
    
    reward_classifier_path: str
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # 1. 从观察中提取图像
        images = extract_images(transition)
        
        # 2. 使用分类器预测成功
        success = self.reward_classifier.predict_reward(
            images, 
            threshold=self.success_threshold
        )
        
        # 3. 根据预测修改奖励
        if success >= self.success_threshold:
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True
        
        # 4. 更新转换
        transition[TransitionKey.REWARD] = reward
        return transition
```

---

## 奖励处理器

### 什么是奖励处理器

奖励处理器是可以在奖励函数计算后修改奖励值的组件。

### 处理器管道

```
环境奖励函数 → 奖励处理器 → 最终奖励值
     ↓              ↓
  reward=0.5    reward=1.0
```

### 使用示例

```python
from lerobot.processor.pipeline import RewardProcessorStep

class ScaleRewardProcessor(RewardProcessorStep):
    """缩放奖励处理器"""
    
    def __init__(self, scale: float = 0.1):
        self.scale = scale
    
    def reward(self, reward):
        """将奖励缩放"""
        return reward * self.scale

# 在配置中使用
processor = ScaleRewardProcessor(scale=0.1)
processed_reward = processor.reward(original_reward)
```

---

## 代码示例

### 示例 1：定义简单的奖励函数

```python
import gym
import numpy as np

class SimpleReachEnv(gym.Env):
    """简单的到达任务环境"""
    
    def __init__(self):
        super().__init__()
        self.goal = np.array([1.0, 1.0])
        self.state = None
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
    
    def reset(self):
        self.state = np.array([0.0, 0.0])
        return self.state
    
    def step(self, action):
        # 执行动作
        self.state = self.state + action * 0.1
        self.state = np.clip(self.state, -2.0, 2.0)
        
        # 计算奖励（奖励函数）
        reward = self._compute_reward(self.state)
        
        # 检查是否完成
        distance = np.linalg.norm(self.state - self.goal)
        done = distance < 0.1
        
        info = {"distance": distance}
        return self.state, reward, done, False, info
    
    def _compute_reward(self, state):
        """奖励函数定义"""
        # 1. 距离目标的负距离（距离越近，奖励越高）
        distance = np.linalg.norm(state - self.goal)
        distance_reward = -distance
        
        # 2. 到达目标的奖励
        if distance < 0.1:
            goal_reward = 10.0
        else:
            goal_reward = 0.0
        
        # 3. 总奖励
        total_reward = distance_reward + goal_reward
        
        return total_reward
```

### 示例 2：记录数据时存储奖励特征

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR, REWARD

# 创建环境
env = SimpleReachEnv()

# 创建数据集
dataset = LeRobotDataset.create(
    repo_id="my_reach_dataset",
    fps=10,
    features={
        "observation.state": {"dtype": "float32", "shape": (2,)},
        "action": {"dtype": "float32", "shape": (2,)},
        "reward": {"dtype": "float32", "shape": (1,)},
        "done": {"dtype": "bool", "shape": (1,)},
    }
)

# 记录数据
obs = env.reset()
for step in range(100):
    # 1. 选择动作（随机或策略）
    action = env.action_space.sample()
    
    # 2. 执行动作，环境计算奖励（奖励函数）
    next_obs, reward, done, truncated, info = env.step(action)
    #    ↑ 奖励函数计算的奖励值
    
    # 3. 构建数据帧
    observation_frame = build_dataset_frame(
        dataset.features,
        {"state": obs},
        prefix=OBS_STR
    )
    action_frame = build_dataset_frame(
        dataset.features,
        action,
        prefix=ACTION
    )
    
    frame = {
        **observation_frame,
        **action_frame,
        "reward": np.array([reward], dtype=np.float32),  # 存储为奖励特征
        "done": np.array([done], dtype=bool),
        "task": "reach_goal"
    }
    
    # 4. 添加到数据集
    dataset.add_frame(frame)
    
    if done:
        dataset.save_episode()
        obs = env.reset()
    else:
        obs = next_obs
```

### 示例 3：从数据集读取奖励特征

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("my_reach_dataset")

# 读取奖励特征
for i in range(len(dataset)):
    sample = dataset[i]
    
    state = sample["observation.state"]
    action = sample["action"]
    reward = sample["reward"]  # 奖励特征（已存储的值）
    done = sample["done"]
    
    print(f"Step {i}: reward={reward.item()}")
```

### 示例 4：使用奖励处理器

```python
from lerobot.processor.pipeline import RewardProcessorStep, DataProcessorPipeline
from lerobot.processor.core import TransitionKey, EnvTransition

class ShapedRewardProcessor(RewardProcessorStep):
    """奖励塑形处理器"""
    
    def reward(self, reward):
        """修改奖励值"""
        # 例如：添加奖励塑形
        shaped_reward = reward + 0.1  # 添加小的正奖励
        return shaped_reward

# 创建处理器管道
reward_processor = ShapedRewardProcessor()

# 在环境步骤中使用
transition = create_transition(
    observation=obs,
    action=action,
    reward=env_reward,  # 环境奖励函数的值
    done=done
)

# 通过处理器修改奖励
processed_transition = reward_processor(transition)
final_reward = processed_transition[TransitionKey.REWARD]
```

---

## 总结

### 奖励函数（Reward Function）

- **定义位置**：环境中
- **计算时机**：运行时（每次 `step()`）
- **输入**：`(state, action, next_state)`
- **输出**：标量奖励值
- **用途**：在线强化学习训练

### 奖励特征（Reward Feature）

- **定义位置**：数据集中
- **存储时机**：记录数据时
- **输入**：无（已经是值）
- **输出**：标量奖励值（已存储）
- **用途**：离线训练、数据分析

### 关键区别

1. **奖励函数**是计算奖励的**函数**，在环境中定义
2. **奖励特征**是存储奖励的**数据**，在数据集中存储
3. **奖励函数**的值会被存储为**奖励特征**

### 最佳实践

1. **定义奖励函数**：在环境的 `step()` 方法中计算奖励
2. **使用奖励处理器**：可以在奖励函数后修改奖励值
3. **存储奖励特征**：记录数据时，将奖励函数的值存储为奖励特征
4. **使用奖励特征**：离线训练时，从数据集中读取奖励特征

---

这些概念对于理解强化学习训练流程非常重要！

