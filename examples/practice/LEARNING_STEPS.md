# 📚 LeRobot 学习步骤完整指南

## 🎯 学习目标

通过系统化的练习，掌握 LeRobot 的完整工作流程：
1. 数据集操作
2. 策略训练
3. 模型评估
4. 真实机器人部署
5. 高级功能

---

## 📖 阶段一：基础入门（1-2天）

### 步骤 1: 环境准备 ✅

**目标**: 安装和配置 LeRobot 环境

```bash
# 1.1 检查 Python 版本（需要 >= 3.10）
python --version

# 1.2 安装 LeRobot
cd /Users/frank/Dev/github/lerobot
pip install -e .

# 1.3 验证安装
python -c "import lerobot; print(f'Version: {lerobot.__version__}')"

# 1.4 安装仿真环境（可选，用于测试）
pip install -e ".[pusht]"
```

**练习任务**:
- [ ] 成功安装 LeRobot
- [ ] 运行 `python examples/practice/quick_start.py` 验证

---

### 步骤 2: 理解数据集格式 📦

**目标**: 学习 LeRobotDataset 的使用

**练习文件**: `examples/dataset/load_lerobot_dataset.py`

```python
# 2.1 查看可用数据集
import lerobot
print(lerobot.available_datasets)

# 2.2 加载数据集元数据（不下载数据）
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
meta = LeRobotDatasetMetadata("lerobot/pusht")
print(f"Episodes: {meta.total_episodes}, FPS: {meta.fps}")

# 2.3 加载完整数据集
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/pusht")

# 2.4 加载特定 episode
dataset = LeRobotDataset("lerobot/pusht", episodes=[0, 1, 2])

# 2.5 使用 delta_timestamps 加载时间序列
delta_timestamps = {
    "observation.image": [-0.1, 0.0],
    "action": [0.0, 0.1, 0.2, 0.3]
}
dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

# 2.6 与 PyTorch DataLoader 配合使用
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**练习任务**:
- [ ] 运行 `examples/dataset/load_lerobot_dataset.py`
- [ ] 尝试加载不同的数据集
- [ ] 理解 delta_timestamps 的作用
- [ ] 使用 DataLoader 批量加载数据

---

### 步骤 3: 可视化数据集 🎨

**目标**: 学会查看和可视化数据集

```bash
# 3.1 使用命令行工具可视化
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0

# 3.2 可视化本地数据集
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --root ./my_data \
    --mode local \
    --episode-index 0
```

**练习任务**:
- [ ] 可视化至少 3 个不同的数据集
- [ ] 理解数据集中的观察和动作格式
- [ ] 查看不同 episode 的数据

---

## 🚀 阶段二：策略训练（3-5天）

### 步骤 4: 训练第一个策略（仿真环境）🎮

**目标**: 在仿真环境中训练策略

**推荐**: 从 PushT 仿真环境开始（无需硬件）

```bash
# 4.1 安装仿真环境
pip install -e ".[pusht]"

# 4.2 使用命令行训练
lerobot-train \
    --config_path=lerobot/diffusion_pusht \
    --output_dir=outputs/my_first_policy

# 4.3 或使用 Python API
python examples/training/train_policy.py
```

**练习文件**: 
- `examples/training/train_policy.py` - 基础训练示例
- `examples/tutorial/diffusion/diffusion_training_example.py` - Diffusion 策略
- `examples/tutorial/act/act_training_example.py` - ACT 策略

**练习任务**:
- [ ] 训练 Diffusion Policy on PushT
- [ ] 训练 ACT Policy on PushT
- [ ] 理解训练过程中的损失函数
- [ ] 查看训练日志和检查点

---

### 步骤 5: 理解不同策略类型 🧠

**目标**: 学习各种策略的特点和适用场景

#### 5.1 ACT (Action Chunking with Transformers)
- **特点**: 使用 Transformer 预测动作序列
- **适用**: 需要动作序列的任务
- **练习**: `examples/tutorial/act/act_training_example.py`

#### 5.2 Diffusion Policy
- **特点**: 使用扩散模型生成动作
- **适用**: 需要平滑动作轨迹的任务
- **练习**: `examples/tutorial/diffusion/diffusion_training_example.py`

#### 5.3 TDMPC
- **特点**: 基于模型的强化学习
- **适用**: 需要探索的任务
- **练习**: 查看文档 `docs/source/policy_tdmpc_README.md`

#### 5.4 VQ-BeT
- **特点**: 向量量化的行为 Transformer
- **适用**: 离散动作空间
- **练习**: 查看文档 `docs/source/policy_vqbet_README.md`

#### 5.5 SmolVLA
- **特点**: 轻量级视觉-语言-动作模型
- **适用**: 需要语言指令的任务
- **练习**: `examples/tutorial/smolvla/using_smolvla_example.py`

**练习任务**:
- [ ] 阅读至少 2 种策略的文档
- [ ] 尝试训练不同的策略
- [ ] 比较不同策略的性能

---

### 步骤 6: 使用预训练模型 🎁

**目标**: 学会使用 Hugging Face Hub 上的预训练模型

```python
# 6.1 加载预训练策略
from lerobot.policies.factory import load_pretrained

policy = load_pretrained("lerobot/diffusion_pusht")

# 6.2 使用策略进行推理
observation = {...}  # 你的观察数据
action = policy.select_action(observation)
```

**练习任务**:
- [ ] 从 Hub 下载预训练模型
- [ ] 使用预训练模型进行推理
- [ ] 理解模型配置和检查点结构

---

## 🤖 阶段三：真实机器人（5-7天）

### 步骤 7: 录制演示数据 📹

**目标**: 使用真实机器人录制训练数据

```bash
# 7.1 基本录制命令
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodemXXX \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0}}" \
    --dataset.repo_id=your_username/your_dataset \
    --dataset.num_episodes=10 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodemYYY
```

**练习文件**: 
- `examples/lekiwi/record.py`
- `examples/phone_to_so100/record.py`
- `examples/so100_to_so100_EE/record.py`

**练习任务**:
- [ ] 录制至少 5 个 episode 的数据
- [ ] 理解数据集的结构
- [ ] 上传数据集到 Hugging Face Hub
- [ ] 验证数据质量

---

### 步骤 8: 训练真实机器人策略 🏋️

**目标**: 使用录制的数据训练策略

```bash
# 8.1 训练策略
lerobot-train \
    --dataset.repo_id=your_username/your_dataset \
    --policy.type=act \
    --output_dir=outputs/train/my_robot_policy \
    --policy.device=cuda \
    --wandb.enable=true
```

**练习任务**:
- [ ] 使用自己的数据训练策略
- [ ] 监控训练过程（使用 WandB）
- [ ] 调整超参数
- [ ] 保存和加载检查点

---

### 步骤 9: 评估和部署 🎯

**目标**: 评估训练好的策略并部署到机器人

```bash
# 9.1 评估策略
lerobot-eval \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodemXXX \
    --policy.path=outputs/train/my_robot_policy/checkpoints/XXXXXX \
    --episodes=10

# 9.2 回放数据集（用于调试）
lerobot-replay \
    --robot.type=so100_follower \
    --dataset.repo_id=your_username/your_dataset \
    --episode-index=0
```

**练习任务**:
- [ ] 评估训练好的策略
- [ ] 分析成功率
- [ ] 调试失败案例
- [ ] 优化策略性能

---

## 🔧 阶段四：高级功能（7-10天）

### 步骤 10: 理解处理器管道 (Processors) ⚙️

**目标**: 学习如何自定义数据处理流程

**文档**: `docs/source/introduction_processors.mdx`

```python
# 10.1 理解三个处理器管道
# - Teleop action processor: 遥操作器动作 → 数据集动作
# - Robot action processor: 数据集动作 → 机器人命令
# - Robot observation processor: 机器人观察 → 数据集观察
```

**练习任务**:
- [ ] 阅读处理器文档
- [ ] 理解绝对 vs 相对末端执行器控制
- [ ] 自定义处理器管道
- [ ] 调试处理器问题

---

### 步骤 11: 强化学习 (RL) 🎲

**目标**: 学习使用 HIL-SERL 进行强化学习

**文档**: `docs/source/hilserl.mdx`

```bash
# 11.1 训练奖励分类器
python examples/tutorial/rl/reward_classifier_example.py

# 11.2 运行 HIL-SERL
python examples/tutorial/rl/hilserl_example.py
```

**练习任务**:
- [ ] 理解 HIL-SERL 工作流程
- [ ] 训练奖励分类器
- [ ] 运行完整的 RL 训练循环
- [ ] 理解人机交互的作用

---

### 步骤 12: 异步推理 🚀

**目标**: 学习使用异步推理提高性能

**文档**: `docs/source/async.mdx`

**练习文件**: 
- `examples/tutorial/async-inf/policy_server.py`
- `examples/tutorial/async-inf/robot_client.py`

**练习任务**:
- [ ] 理解异步推理架构
- [ ] 设置策略服务器
- [ ] 配置机器人客户端
- [ ] 测量性能提升

---

### 步骤 13: 多 GPU 训练 💪

**目标**: 学习使用多 GPU 加速训练

**文档**: `docs/source/multi_gpu_training.mdx`

```bash
# 13.1 使用 accelerate 进行多 GPU 训练
accelerate launch lerobot-train \
    --dataset.repo_id=your_dataset \
    --policy.type=act \
    --output_dir=outputs/multi_gpu
```

**练习任务**:
- [ ] 配置多 GPU 环境
- [ ] 使用 accelerate 训练
- [ ] 监控 GPU 使用率
- [ ] 优化训练速度

---

### 步骤 14: 自定义硬件集成 🔌

**目标**: 集成自己的机器人硬件

**文档**: `docs/source/integrate_hardware.mdx`

**练习任务**:
- [ ] 阅读硬件集成指南
- [ ] 实现 Robot 接口
- [ ] 实现 Teleoperator 接口
- [ ] 测试硬件连接

---

## 📊 阶段五：项目实践（10+天）

### 步骤 15: 完整项目实践 🎓

**目标**: 完成一个端到端的机器人学习项目

**项目建议**:
1. **简单任务**: 抓取和放置物体
2. **中等任务**: 多步骤操作（如打开抽屉、取出物品）
3. **复杂任务**: 需要语言指令的任务

**项目检查清单**:
- [ ] 定义任务和目标
- [ ] 录制至少 50 个 episode 的数据
- [ ] 训练和调优策略
- [ ] 评估性能（成功率 > 80%）
- [ ] 记录实验过程和结果
- [ ] 分享到 Hugging Face Hub

---

## 📝 学习检查清单

### 基础技能 ✅
- [ ] 能够安装和配置 LeRobot
- [ ] 理解数据集格式和加载方法
- [ ] 能够可视化数据集
- [ ] 能够在仿真环境中训练策略

### 中级技能 🎯
- [ ] 理解至少 2 种策略类型
- [ ] 能够录制真实机器人数据
- [ ] 能够训练真实机器人策略
- [ ] 能够评估和部署策略

### 高级技能 🚀
- [ ] 能够自定义处理器管道
- [ ] 能够使用强化学习
- [ ] 能够使用异步推理
- [ ] 能够集成自定义硬件

---

## 🎯 推荐学习路径

### 路径 A: 纯软件学习（无硬件）
1. 步骤 1-3: 基础入门
2. 步骤 4-6: 策略训练
3. 步骤 10-13: 高级功能
4. 步骤 15: 仿真项目实践

### 路径 B: 真实机器人学习（有硬件）
1. 步骤 1-3: 基础入门
2. 步骤 4-6: 策略训练（先用仿真）
3. 步骤 7-9: 真实机器人
4. 步骤 10-14: 高级功能
5. 步骤 15: 完整项目

### 路径 C: 快速体验（1天）
1. 步骤 1: 环境准备
2. 步骤 2: 数据集操作
3. 步骤 4: 训练仿真策略
4. 步骤 6: 使用预训练模型

---

## 📚 学习资源

- 📖 **官方文档**: https://huggingface.co/docs/lerobot
- 💬 **Discord 社区**: https://discord.gg/s3KuuzsPFb
- 🐛 **GitHub Issues**: https://github.com/huggingface/lerobot/issues
- 📦 **数据集 Hub**: https://huggingface.co/lerobot
- 🎥 **视频教程**: 查看文档中的视频链接

---

## 💡 学习建议

1. **循序渐进**: 不要跳过基础步骤
2. **动手实践**: 每个步骤都要实际运行代码
3. **记录笔记**: 记录遇到的问题和解决方案
4. **参与社区**: 在 Discord 中提问和分享
5. **持续学习**: 关注项目更新和新功能

---

**祝你学习愉快！🤖✨**

