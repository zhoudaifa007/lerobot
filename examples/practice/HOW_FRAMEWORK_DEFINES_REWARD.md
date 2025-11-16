# LeRobot 框架如何定义奖励函数

本文档详细说明 LeRobot 框架中奖励函数的定义机制和使用方式。

## 📋 目录

1. [框架的奖励函数机制](#框架的奖励函数机制)
2. [奖励函数的来源](#奖励函数的来源)
3. [奖励处理流程](#奖励处理流程)
4. [如何自定义奖励函数](#如何自定义奖励函数)
5. [代码示例](#代码示例)

---

## 框架的奖励函数机制

### 核心原则

LeRobot 框架采用**分层奖励机制**：

1. **环境层**：底层环境（如 Gym、Metaworld、LIBERO）计算基础奖励
2. **处理器层**：奖励处理器可以修改或增强奖励
3. **最终奖励**：环境奖励 + 处理器奖励

### 奖励计算流程

```
┌─────────────────┐
│   环境 (Env)    │
│  env.step()     │
│       ↓         │
│  基础奖励计算    │
│  reward_env     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  动作处理器      │
│  action_processor│
│       ↓         │
│  处理器奖励      │
│  reward_proc    │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  最终奖励        │
│  reward =        │
│  reward_env +    │
│  reward_proc     │
└─────────────────┘
```

---

## 奖励函数的来源

### 1. 环境内置奖励函数

大多数环境（如 Metaworld、LIBERO）在内部定义了奖励函数：

```python
# src/lerobot/envs/metaworld.py
class MetaworldEnv(gym.Env):
    def step(self, action):
        # 底层环境计算奖励（奖励函数在环境内部）
        raw_obs, reward, done, truncated, info = self._env.step(action)
        #    ↑ 奖励由底层 Metaworld 环境计算
        
        return observation, reward, terminated, truncated, info
```

```python
# src/lerobot/envs/libero.py
class LiberoEnv(gym.Env):
    def step(self, action):
        # 底层环境计算奖励（奖励函数在环境内部）
        raw_obs, reward, done, info = self._env.step(action)
        #    ↑ 奖励由底层 LIBERO 环境计算
        
        return observation, reward, terminated, truncated, info
```

### 2. 机器人环境（零奖励）

对于真实机器人环境，默认返回零奖励：

```python
# src/lerobot/rl/gym_manipulator.py
class RobotEnv(gym.Env):
    def step(self, action):
        # 执行动作
        self.robot.send_action(joint_targets_dict)
        obs = self._get_observation()
        
        # 默认奖励为 0（需要外部定义奖励函数）
        reward = 0.0
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, info
```

### 3. 处理器奖励

奖励可以通过处理器添加或修改：

```python
# src/lerobot/rl/gym_manipulator.py
def step_env_and_process_transition(...):
    # 1. 环境计算奖励
    obs, reward, terminated, truncated, info = env.step(processed_action)
    #    ↑ 环境奖励
    
    # 2. 处理器可能添加额外奖励
    reward = reward + processed_action_transition[TransitionKey.REWARD]
    #    ↑ 环境奖励 + 处理器奖励
```

---

## 奖励处理流程

### 完整流程

```python
# src/lerobot/rl/gym_manipulator.py
def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline,
    action_processor: DataProcessorPipeline,
) -> EnvTransition:
    # 1. 创建动作转换
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = env.get_raw_joint_positions()
    
    # 2. 通过动作处理器处理（可能添加奖励）
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]
    
    # 3. 环境执行步骤（环境计算奖励）
    obs, reward, terminated, truncated, info = env.step(processed_action)
    #    ↑ 环境奖励函数计算的奖励
    
    # 4. 合并奖励（环境奖励 + 处理器奖励）
    reward = reward + processed_action_transition[TransitionKey.REWARD]
    #    ↑ 最终奖励 = 环境奖励 + 处理器奖励
    
    # 5. 创建新转换
    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,  # 最终奖励
        done=terminated,
        truncated=truncated,
        info=info,
    )
    
    # 6. 通过环境处理器处理（可能修改奖励）
    new_transition = env_processor(new_transition)
    
    return new_transition
```

### 关键点

1. **环境奖励**：由 `env.step()` 返回
2. **处理器奖励**：由 `action_processor` 在 `TransitionKey.REWARD` 中添加
3. **最终奖励**：`环境奖励 + 处理器奖励`
4. **后处理**：`env_processor` 可以进一步修改奖励

---

## 如何自定义奖励函数

### 方法 1：在自定义环境中定义

```python
import gymnasium as gym
import numpy as np

class CustomRobotEnv(gym.Env):
    """自定义机器人环境，包含奖励函数"""
    
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
        
        return self.state, reward, done, False, {}
    
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
        
        # 3. 动作平滑性奖励
        action_penalty = -0.1 * np.linalg.norm(action)
        
        # 总奖励
        total_reward = distance_reward + completion_reward + action_penalty
        
        return total_reward
```

### 方法 2：使用奖励处理器

```python
from lerobot.processor.pipeline import RewardProcessorStep
from lerobot.processor.core import TransitionKey, EnvTransition

class DistanceRewardProcessor(RewardProcessorStep):
    """基于距离的奖励处理器"""
    
    def __init__(self, goal, distance_scale=1.0):
        self.goal = goal
        self.distance_scale = distance_scale
    
    def reward(self, reward):
        """修改奖励值"""
        # 从转换中获取状态（需要访问当前转换）
        transition = self._current_transition
        observation = transition.get(TransitionKey.OBSERVATION)
        
        if observation is None:
            return reward
        
        # 计算距离奖励
        state = observation.get("state", None)
        if state is not None:
            distance = np.linalg.norm(state - self.goal)
            distance_reward = -distance * self.distance_scale
            return reward + distance_reward
        
        return reward
```

### 方法 3：使用奖励分类器

LeRobot 提供了 `RewardClassifierProcessorStep`，可以根据图像分类器预测成功：

```python
# 在配置中使用
@dataclass
class RewardClassifierProcessorStep(ProcessorStep):
    """使用奖励分类器修改奖励"""
    
    pretrained_path: str
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

### 方法 4：在动作处理器中添加奖励

```python
from lerobot.processor.pipeline import ActionProcessorStep
from lerobot.processor.core import TransitionKey

class PenaltyActionProcessor(ActionProcessorStep):
    """动作惩罚处理器"""
    
    def __init__(self, penalty_scale=0.01):
        self.penalty_scale = penalty_scale
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition[TransitionKey.ACTION]
        
        # 计算动作惩罚（大动作惩罚）
        action_magnitude = torch.norm(action)
        penalty = -self.penalty_scale * action_magnitude
        
        # 添加到转换的奖励中
        current_reward = new_transition.get(TransitionKey.REWARD, 0.0)
        new_transition[TransitionKey.REWARD] = current_reward + penalty
        
        return new_transition
```

---

## 代码示例

### 示例 1：使用环境内置奖励函数

```python
from lerobot.envs.metaworld import MetaworldEnv

# 创建环境（奖励函数在环境内部）
env = MetaworldEnv(task="metaworld-reach-v2")

# 执行步骤（环境自动计算奖励）
obs, reward, done, truncated, info = env.step(action)
#    ↑ 奖励由 Metaworld 环境的奖励函数计算
```

### 示例 2：自定义环境奖励函数

```python
import gymnasium as gym
import numpy as np

class MyCustomEnv(gym.Env):
    def step(self, action):
        # 执行动作
        next_state = self._execute_action(action)
        
        # 自定义奖励函数
        reward = self._compute_reward(self.state, action, next_state)
        
        return next_state, reward, done, info
    
    def _compute_reward(self, state, action, next_state):
        """自定义奖励函数"""
        # 奖励逻辑
        distance = np.linalg.norm(next_state - self.goal)
        reward = -distance
        
        if distance < 0.1:
            reward += 10.0
        
        return reward
```

### 示例 3：使用奖励处理器修改奖励

```python
from lerobot.processor.pipeline import RewardProcessorStep
from lerobot.processor.core import TransitionKey

class ScaleRewardProcessor(RewardProcessorStep):
    """缩放奖励处理器"""
    
    def __init__(self, scale: float = 0.1):
        self.scale = scale
    
    def reward(self, reward):
        """缩放奖励"""
        return reward * self.scale

# 在处理器管道中使用
processor = ScaleRewardProcessor(scale=0.1)
processed_reward = processor.reward(original_reward)
```

### 示例 4：在动作处理器中添加奖励

```python
from lerobot.processor.pipeline import ActionProcessorStep
from lerobot.processor.core import TransitionKey
import torch

class SmoothActionProcessor(ActionProcessorStep):
    """平滑动作奖励处理器"""
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition[TransitionKey.ACTION]
        
        # 计算平滑性奖励（惩罚大动作）
        action_magnitude = torch.norm(action)
        smoothness_reward = -0.01 * action_magnitude
        
        # 添加到转换的奖励中
        current_reward = new_transition.get(TransitionKey.REWARD, 0.0)
        new_transition[TransitionKey.REWARD] = current_reward + smoothness_reward
        
        return new_transition
```

### 示例 5：完整的奖励处理流程

```python
from lerobot.rl.gym_manipulator import step_env_and_process_transition
from lerobot.processor.pipeline import DataProcessorPipeline

# 创建环境
env = MyCustomEnv()

# 创建处理器
action_processor = DataProcessorPipeline([
    SmoothActionProcessor(),  # 添加平滑性奖励
])

env_processor = DataProcessorPipeline([
    ScaleRewardProcessor(scale=0.1),  # 缩放奖励
])

# 执行步骤
transition = create_transition(...)
new_transition = step_env_and_process_transition(
    env=env,
    transition=transition,
    action=action,
    env_processor=env_processor,
    action_processor=action_processor,
)

# 最终奖励 = 环境奖励 + 动作处理器奖励，然后通过环境处理器缩放
final_reward = new_transition[TransitionKey.REWARD]
```

---

## 框架的奖励函数总结

### 奖励函数的定义位置

1. **环境内部**（最常见）
   - Metaworld、LIBERO 等环境在内部定义奖励函数
   - 通过 `env.step()` 返回奖励值

2. **自定义环境**
   - 在 `step()` 方法中实现 `_compute_reward()` 方法
   - 根据状态、动作、下一个状态计算奖励

3. **奖励处理器**
   - 通过 `RewardProcessorStep` 修改奖励
   - 通过 `ActionProcessorStep` 添加奖励

### 奖励计算流程

```
环境 step() → 环境奖励
     ↓
动作处理器 → 处理器奖励
     ↓
最终奖励 = 环境奖励 + 处理器奖励
     ↓
环境处理器 → 可能进一步修改奖励
```

### 关键代码位置

1. **环境奖励**：
   - `src/lerobot/envs/metaworld.py` - MetaworldEnv.step()
   - `src/lerobot/envs/libero.py` - LiberoEnv.step()
   - `src/lerobot/rl/gym_manipulator.py` - RobotEnv.step()

2. **奖励处理**：
   - `src/lerobot/rl/gym_manipulator.py` - step_env_and_process_transition()
   - `src/lerobot/processor/pipeline.py` - RewardProcessorStep
   - `src/lerobot/processor/hil_processor.py` - RewardClassifierProcessorStep

### 最佳实践

1. **使用环境内置奖励**：对于标准环境（Metaworld、LIBERO），使用环境内置的奖励函数
2. **自定义环境奖励**：对于自定义任务，在环境的 `step()` 方法中定义奖励函数
3. **使用处理器增强**：通过奖励处理器添加额外的奖励信号（如平滑性、安全性）
4. **奖励分类器**：对于难以定义奖励的任务，使用奖励分类器从图像预测成功

---

这些机制使得 LeRobot 框架能够灵活地处理各种奖励函数定义方式！

