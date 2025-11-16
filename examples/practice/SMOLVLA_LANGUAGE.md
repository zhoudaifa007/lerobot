# SmolVLA 语言功能说明

本文档说明 SmolVLA 的语言功能，以及如何使用语言指令。

## 📋 核心答案

**SmolVLA 已经内置了语言功能！**

SmolVLA 本身就是 **Vision-Language-Action (VLA)** 模型，**Language（语言）是它的核心组成部分之一**，不需要额外添加。

---

## 🎯 SmolVLA 的三模态架构

### 输入模态

SmolVLA 接收**三种输入**：

1. **Vision（视觉）**：多个摄像头视图
2. **Language（语言）**：自然语言指令 ✅ **已内置**
3. **State（状态）**：机器人当前传感器状态

### 输出

- **Action（动作）**：机器人控制序列

---

## 💻 语言功能的实现

### 1. 语言输入处理

在代码中，语言功能已经完整实现：

```python
# src/lerobot/policies/smolvla/modeling_smolvla.py (260-263)
lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

actions = self.model.sample_actions(
    images, img_masks, 
    lang_tokens, lang_masks,  # ← 语言输入
    state, noise=noise
)
```

### 2. 语言预处理流程

在预处理管道中，语言指令会被：

```python
# src/lerobot/policies/smolvla/processor_smolvla.py (53-78)
# 4. 确保语言任务描述以换行符结尾
SmolVLANewLineProcessor(),

# 5. 对语言任务描述进行分词
TokenizerProcessorStep(
    tokenizer_name=config.vlm_model_name,
    padding=config.pad_language_to,
    padding_side="right",
    max_length=config.tokenizer_max_length,  # 默认 48 tokens
),
```

### 3. VLM 骨干网络

SmolVLA 使用 **Vision-Language Model (VLM)** 作为骨干网络：

```python
# src/lerobot/policies/smolvla/configuration_smolvla.py (86)
vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
```

这个 VLM 模型本身就支持：
- ✅ 视觉理解
- ✅ 语言理解
- ✅ 视觉-语言融合

---

## 🚀 如何使用语言功能

### 1. 在数据收集中添加语言指令

在录制数据时，使用 `--dataset.single_task` 参数指定任务描述：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --dataset.single_task="Grasp a lego block and put it in the bin." \  # ← 语言指令
  --dataset.repo_id=${HF_USER}/my_dataset \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10
```

### 2. 在评估时使用语言指令

在评估模型时，使用相同的任务描述：

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --dataset.single_task="Grasp a lego block and put it in the bin." \  # ← 使用相同的语言指令
  --dataset.repo_id=${HF_USER}/eval_dataset \
  --policy.path=HF_USER/FINETUNE_MODEL_NAME
```

### 3. 语言指令的要求

根据文档说明：

```python
# src/lerobot/policies/smolvla/processor_smolvla.py (53)
# 4. Ensuring the language task description ends with a newline character.
```

**重要**：语言指令应该以换行符结尾（`\n`），但 `SmolVLANewLineProcessor` 会自动处理这一点。

---

## 📊 语言功能的配置

### Tokenizer 配置

```python
# src/lerobot/policies/smolvla/configuration_smolvla.py (61-62)
tokenizer_max_length: int = 48  # 最大 token 长度
```

### 语言填充配置

```python
# src/lerobot/policies/smolvla/configuration_smolvla.py (95)
pad_language_to: str = "longest"  # 或 "max_length"
```

### VLM 模型配置

```python
# src/lerobot/policies/smolvla/configuration_smolvla.py (86)
vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
```

---

## 🔍 语言功能的工作流程

### 完整流程

```
1. 输入自然语言指令
   ↓
2. SmolVLANewLineProcessor：确保以换行符结尾
   ↓
3. TokenizerProcessorStep：将文本转换为 tokens
   ↓
4. 语言 tokens 与视觉、状态特征融合
   ↓
5. VLM 模型处理多模态输入
   ↓
6. Action Expert 生成动作序列
   ↓
7. 输出机器人控制动作
```

### 代码实现

```python
# src/lerobot/policies/smolvla/modeling_smolvla.py (671-686)
def forward(
    self, images, img_masks, 
    lang_tokens, lang_masks,  # ← 语言输入
    state, actions, noise=None, time=None
):
    # 嵌入前缀（视觉 + 语言 + 状态）
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, 
        lang_tokens, lang_masks,  # ← 语言 tokens
        state=state
    )
    
    # 嵌入后缀（动作）
    suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
    
    # 融合多模态特征
    # ...
```

---

## 📝 使用示例

### 示例 1：基本使用

```bash
# 录制数据时指定任务
lerobot-record \
  --dataset.single_task="Pick up the red cube and place it in the blue box." \
  --dataset.repo_id=my_user/my_task_dataset

# 训练模型
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=my_user/my_task_dataset \
  --batch_size=64 \
  --steps=20000

# 评估模型（使用相同的任务描述）
lerobot-record \
  --dataset.single_task="Pick up the red cube and place it in the blue box." \
  --policy.path=my_user/my_trained_model
```

### 示例 2：不同任务描述

```bash
# 任务 1：抓取任务
--dataset.single_task="Grasp the lego block."

# 任务 2：放置任务
--dataset.single_task="Put the cube in the bin."

# 任务 3：复杂任务
--dataset.single_task="Pick up the red cube, move it to the left, and place it in the blue container."
```

---

## ⚙️ 语言功能的优势

### 1. 多任务支持

通过不同的语言指令，同一个模型可以执行不同的任务：

- "Pick up the cube"
- "Place the cube in the bin"
- "Move the object to the left"

### 2. 零样本泛化

训练后的模型可以理解新的语言指令，即使这些指令在训练时没有完全相同的表述。

### 3. 自然交互

用户可以使用自然语言描述任务，无需编程或重新训练模型。

---

## 🔧 高级配置

### 自定义 Tokenizer

```python
# 在配置中修改
config = SmolVLAConfig(
    vlm_model_name="your-custom-vlm-model",  # 使用自定义 VLM
    tokenizer_max_length=64,  # 增加最大长度
    pad_language_to="max_length",  # 使用固定长度填充
)
```

### 语言指令格式

语言指令应该：
- ✅ 清晰描述任务目标
- ✅ 使用自然语言
- ✅ 与训练时的格式一致
- ✅ 以换行符结尾（自动处理）

---

## 📊 语言功能的技术细节

### 1. 语言嵌入

```python
# src/lerobot/policies/smolvla/smolvlm_with_expert.py (195)
def embed_language_tokens(self, tokens: torch.Tensor):
    return self.get_vlm_model().text_model.get_input_embeddings()(tokens)
```

### 2. 多模态融合

语言 tokens 与视觉特征和状态特征在 VLM 模型中进行融合：

```python
# 前缀嵌入（视觉 + 语言 + 状态）
prefix_embs = embed_prefix(images, lang_tokens, state)

# 后缀嵌入（动作）
suffix_embs = embed_suffix(actions)

# 融合
output = vlm_with_expert.forward(
    inputs_embeds=[prefix_embs, suffix_embs],
    attention_mask=att_masks,
    ...
)
```

### 3. 注意力机制

语言 tokens 通过交叉注意力机制与视觉和动作特征交互：

```python
# 注意力掩码包含语言 tokens
att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
```

---

## ❓ 常见问题

### Q1: 我需要单独添加语言功能吗？

**A**: 不需要！SmolVLA 已经内置了完整的语言功能。

### Q2: 如何提供语言指令？

**A**: 在数据收集和评估时，使用 `--dataset.single_task` 参数：

```bash
--dataset.single_task="Your task description here."
```

### Q3: 语言指令的长度有限制吗？

**A**: 是的，默认最大长度为 48 tokens。可以通过配置修改：

```python
tokenizer_max_length: int = 64  # 增加到 64 tokens
```

### Q4: 可以使用不同的语言吗？

**A**: 这取决于 VLM 模型的支持。默认的 `SmolVLM2-500M-Video-Instruct` 主要支持英语，但可以尝试其他语言。

### Q5: 语言指令必须与训练时完全一致吗？

**A**: 不需要完全一致，但应该：
- 使用相似的任务描述格式
- 描述相同的任务类型
- 使用自然语言

---

## 📝 总结

### 核心要点

1. **SmolVLA 已经内置语言功能** ✅
   - Language 是 VLA 模型的核心组成部分
   - 不需要额外添加

2. **使用方法**
   - 在数据收集时：`--dataset.single_task="任务描述"`
   - 在评估时：使用相同的任务描述

3. **技术实现**
   - 使用 VLM 骨干网络处理语言
   - 通过 Tokenizer 将文本转换为 tokens
   - 与视觉和状态特征融合

4. **优势**
   - 多任务支持
   - 零样本泛化
   - 自然语言交互

### 关键代码位置

- **语言处理**：`src/lerobot/policies/smolvla/processor_smolvla.py`
- **语言嵌入**：`src/lerobot/policies/smolvla/smolvlm_with_expert.py`
- **模型前向传播**：`src/lerobot/policies/smolvla/modeling_smolvla.py`

---

**SmolVLA 的语言功能已经完全集成，开箱即用！** 🎉

