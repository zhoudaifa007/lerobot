# SmolVLA policy.train() 调用链分析

本文档分析当 policy 设置为 SmolVLA 时，`policy.train()` 的调用链。

## 📋 调用链

当在 `lerobot_train.py` 中调用 `policy.train()` 时，调用链如下：

```
lerobot_train.py (302)
    ↓
policy.train()  # SmolVLAPolicy 实例
    ↓
nn.Module.train()  # PyTorch 基类方法
    ↓
递归调用所有子模块的 train()
    ├─→ self.model.train()  # VLAFlowMatching
    │       ↓
    │   nn.Module.train()  # VLAFlowMatching 继承自 nn.Module，**没有重写 train()**
    │       ↓
    │   使用 PyTorch 默认实现：递归调用所有子模块的 train()
    │       ├─→ self.vlm_with_expert.train()  # SmolVLMWithExpertModel
    │       │       ↓
    │       │   SmolVLMWithExpertModel.train()  # 重写的 train() 方法
    │       │       ↓
    │       │   super().train(mode)  # 调用 nn.Module.train()
    │       │       ↓
    │       │   然后设置特定模块为 eval 模式
    │       │       ├─→ 如果 freeze_vision_encoder: vision_model.eval()
    │       │       └─→ 如果 train_expert_only: vlm.eval()
    │       │
    │       ├─→ self.state_proj.train()
    │       ├─→ self.action_in_proj.train()
    │       ├─→ self.action_out_proj.train()
    │       └─→ self.action_time_mlp_*.train()
    │
    └─→ 其他子模块的 train()
```

---

## 🔍 详细分析

### 1. 初始调用

```302:302:src/lerobot/scripts/lerobot_train.py
    policy.train()
```

**`policy` 是 `SmolVLAPolicy` 实例**：
```216:237:src/lerobot/policies/smolvla/modeling_smolvla.py
class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"

    def __init__(
        self,
        config: SmolVLAConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = VLAFlowMatching(config)
        self.reset()
```

### 2. 继承关系

**`SmolVLAPolicy` 继承链**：
```
SmolVLAPolicy
    ↓ 继承自
PreTrainedPolicy
    ↓ 继承自
nn.Module (PyTorch)
```

**`SmolVLAPolicy` 没有重写 `train()` 方法**，所以会使用 `nn.Module.train()`。

### 3. PyTorch 的 train() 方法

**`nn.Module.train()` 的行为**：
- 设置模块为训练模式（`self.training = True`）
- 递归调用所有子模块的 `train()` 方法
- 启用 Dropout、BatchNorm 等训练时特性

### 4. 关键子模块：VLAFlowMatching

**重要**：`VLAFlowMatching` **没有重写 `train()` 方法**，它继承自 `nn.Module`，所以直接使用 PyTorch 的默认 `train()` 方法，该方法会递归调用所有子模块的 `train()`。

**`self.model` 是 `VLAFlowMatching` 实例**：
```448:512:src/lerobot/policies/smolvla/modeling_smolvla.py
class VLAFlowMatching(nn.Module):
    """
    SmolVLA
    ...
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device,
        )
        self.state_proj = nn.Linear(...)
        self.action_in_proj = nn.Linear(...)
        self.action_out_proj = nn.Linear(...)
        self.action_time_mlp_in = nn.Linear(...)
        self.action_time_mlp_out = nn.Linear(...)
        ...
```

**注意**：`VLAFlowMatching` 类**没有定义 `train()` 方法**，所以它使用 `nn.Module` 的默认 `train()` 方法，该方法会递归调用所有子模块（如 `vlm_with_expert`、`state_proj` 等）的 `train()` 方法。

### 5. 关键子模块：SmolVLMWithExpertModel

**`self.vlm_with_expert` 重写了 `train()` 方法**：
```171:178:src/lerobot/policies/smolvla/smolvlm_with_expert.py
    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()
```

**这是关键的调用点！**

---

## 🎯 关键调用点

### SmolVLMWithExpertModel.train()

**位置**：`src/lerobot/policies/smolvla/smolvlm_with_expert.py` (171-178)

**作用**：
1. 调用父类的 `train()` 方法（设置所有模块为训练模式）
2. **然后**根据配置，将某些模块设置为 `eval()` 模式：
   - 如果 `freeze_vision_encoder=True`：冻结视觉编码器
   - 如果 `train_expert_only=True`：冻结整个 VLM，只训练 Action Expert

**默认配置**：
```70:73:src/lerobot/policies/smolvla/configuration_smolvla.py
    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True
```

**说明**：
- 默认情况下，**只训练 Action Expert**
- VLM 部分保持 `eval()` 模式（冻结）

---

## 📊 完整调用流程

### 步骤 1：调用 policy.train()

```python
# lerobot_train.py (302)
policy.train()  # policy 是 SmolVLAPolicy 实例
```

### 步骤 2：PyTorch 递归调用

```python
# nn.Module.train() (PyTorch 内部)
# 1. 设置 self.training = True
# 2. 递归调用所有子模块的 train()
```

### 步骤 3：调用 VLAFlowMatching.train()

**重要**：`VLAFlowMatching` **没有重写 `train()` 方法**，它继承自 `nn.Module`，所以使用 PyTorch 的默认 `train()` 方法。

```python
# self.model.train()  # VLAFlowMatching
# 使用 nn.Module.train() (PyTorch 默认实现)
# 1. 设置 self.training = True
# 2. 递归调用所有子模块的 train()
```

### 步骤 4：调用 SmolVLMWithExpertModel.train()

```171:178:src/lerobot/policies/smolvla/smolvlm_with_expert.py
    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()
```

**这是实际执行逻辑的地方！**

### 步骤 5：设置特定模块为 eval 模式

根据配置：
- **Action Expert**：保持 `train()` 模式（可训练）
- **VLM**：设置为 `eval()` 模式（冻结）
- **Vision Encoder**：设置为 `eval()` 模式（冻结）

---

## 🔧 实际效果

### 默认配置下的行为

```python
# 默认配置
freeze_vision_encoder: bool = True
train_expert_only: bool = True
```

**调用 `policy.train()` 后**：

| 模块 | 模式 | 可训练 |
|------|------|--------|
| **Action Expert** | `train()` | ✅ 是 |
| **State Projection** | `train()` | ✅ 是 |
| **VLM** | `eval()` | ❌ 否（冻结） |
| **Vision Encoder** | `eval()` | ❌ 否（冻结） |

### 代码验证

```139:147:src/lerobot/policies/smolvla/smolvlm_with_expert.py
    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
```

**`set_requires_grad()` 在初始化时调用**，确保参数被冻结。

**`train()` 方法确保在训练时，这些模块保持 `eval()` 模式**。

---

## 💡 为什么需要重写 train()？

### 问题

PyTorch 的 `nn.Module.train()` 会将**所有**子模块设置为训练模式。但对于 SmolVLA：
- 需要保持某些模块在 `eval()` 模式（冻结）
- 即使调用 `train()`，也要确保这些模块保持冻结

### 解决方案

重写 `train()` 方法：
1. 先调用 `super().train(mode)` 设置所有模块为训练模式
2. 然后根据配置，将特定模块设置回 `eval()` 模式

---

## 📝 代码位置总结

### 调用链

```
1. lerobot_train.py (302)
   └─→ policy.train()

2. nn.Module.train() (PyTorch)
   └─→ 递归调用所有子模块

3. VLAFlowMatching.train() (继承自 nn.Module，**没有重写**)
   └─→ 使用 PyTorch 默认的 train() 方法
       └─→ 递归调用所有子模块的 train()

4. SmolVLMWithExpertModel.train() ⭐ 关键调用点
   └─→ src/lerobot/policies/smolvla/smolvlm_with_expert.py (171-178)
       ├─→ super().train(mode)
       ├─→ 如果 freeze_vision_encoder: vision_model.eval()
       └─→ 如果 train_expert_only: vlm.eval()
```

---

## 🎯 总结

### 核心答案

**当 policy 设置为 SmolVLA 时，`policy.train()` 最终会调用到：**

**`SmolVLMWithExpertModel.train()`** 方法

**位置**：`src/lerobot/policies/smolvla/smolvlm_with_expert.py` (171-178)

### 关键行为

1. **设置所有模块为训练模式**：调用 `super().train(mode)`
2. **冻结 VLM**：如果 `train_expert_only=True`，将 VLM 设置为 `eval()` 模式
3. **冻结视觉编码器**：如果 `freeze_vision_encoder=True`，将视觉编码器设置为 `eval()` 模式
4. **保持 Action Expert 可训练**：Action Expert 保持 `train()` 模式

### 默认效果

- ✅ **Action Expert**：可训练
- ❌ **VLM**：冻结（eval 模式）
- ❌ **Vision Encoder**：冻结（eval 模式）

---

**`policy.train()` 最终调用到 `SmolVLMWithExpertModel.train()`，确保只训练 Action Expert，而 VLM 和视觉编码器保持冻结！** 🎯

