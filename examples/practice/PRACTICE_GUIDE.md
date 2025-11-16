# LeRobot 实践指南

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保 Python 版本 >= 3.10
python --version

# 安装 LeRobot（如果还没安装）
pip install lerobot

# 或者从源码安装（推荐用于开发）
pip install -e .
```

### 2. 运行快速入门示例

```bash
cd examples/practice
python quick_start.py
```

或者从项目根目录运行：

```bash
python examples/practice/quick_start.py
```

这个脚本会：
- 显示可用的数据集
- 加载一个示例数据集（PushT）
- 显示数据集的基本信息
- 查看第一帧数据

### 3. 实践路径

#### 路径 A: 使用仿真环境（推荐新手）

```bash
# 安装仿真环境
pip install -e ".[pusht]"

# 运行数据集加载示例
python examples/dataset/load_lerobot_dataset.py

# 运行训练示例（需要 GPU 或较长时间）
python examples/training/train_policy.py
```

#### 路径 B: 使用真实机器人

如果你有支持的机器人硬件：

```bash
# 1. 录制数据
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodemXXX \
    --dataset.repo_id=your_username/your_dataset \
    --dataset.num_episodes=5

# 2. 训练策略
lerobot-train \
    --dataset.repo_id=your_username/your_dataset \
    --policy.type=act \
    --output_dir=outputs/train/my_policy

# 3. 评估策略
lerobot-eval \
    --robot.type=so100_follower \
    --policy.path=outputs/train/my_policy/checkpoints/XXXXXX
```

### 4. 可用的命令行工具

LeRobot 提供了多个命令行工具：

- `lerobot-record` - 录制机器人演示数据
- `lerobot-train` - 训练策略模型
- `lerobot-eval` - 评估训练好的策略
- `lerobot-replay` - 回放数据集中的动作
- `lerobot-dataset-viz` - 可视化数据集
- `lerobot-info` - 显示系统信息

查看帮助：
```bash
lerobot-train --help
lerobot-record --help
```

### 5. 示例代码位置

- `examples/dataset/` - 数据集使用示例
- `examples/training/` - 训练示例
- `examples/tutorial/` - 各种策略的教程
- `examples/lekiwi/` - LeKiwi 机器人完整示例
- `examples/phone_to_so100/` - 手机控制 SO-100 示例

### 6. 常用数据集

- `lerobot/pusht` - PushT 仿真环境（小数据集，适合测试）
- `lerobot/aloha_mobile_cabinet` - ALOHA 机器人数据集
- 更多数据集: https://huggingface.co/datasets?other=LeRobot

### 7. 支持的策略类型

- **ACT** - Action Chunking with Transformers
- **Diffusion** - Diffusion Policy
- **TDMPC** - TD-MPC
- **VQ-BeT** - Vector Quantized Behavior Transformer
- **SmolVLA** - Small Vision-Language-Action model

### 8. 获取帮助

- 📚 文档: https://huggingface.co/docs/lerobot
- 💬 Discord: https://discord.gg/s3KuuzsPFb
- 🐛 Issues: https://github.com/huggingface/lerobot/issues

### 9. 下一步

1. ✅ 运行 `python quick_start.py` 验证安装
2. 📖 阅读 `examples/` 目录中的示例代码
3. 🎯 选择一个简单的任务开始（推荐 PushT 仿真环境）
4. 🤖 如果有硬件，尝试录制和训练自己的数据

### 10. 故障排除

**问题：无法下载数据集**
- 检查网络连接
- 运行 `huggingface-cli login` 登录

**问题：CUDA/GPU 相关错误**
- 检查 PyTorch 是否正确安装 GPU 版本
- 使用 `--policy.device=cpu` 在 CPU 上运行

**问题：导入错误**
- 确保在正确的虚拟环境中
- 运行 `pip install -e .` 重新安装

