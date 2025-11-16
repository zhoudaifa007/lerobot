# 🎯 下一步行动指南

## 当前状态
- ✅ 已创建实践示例文件
- ❌ LeRobot 尚未安装
- ✅ Python 3.12.11 已就绪

## 📋 推荐步骤

### 步骤 1: 安装 LeRobot

你有两个选择：

#### 选项 A: 从源码安装（推荐，用于开发）
```bash
cd /Users/frank/Dev/github/lerobot
pip install -e .
```

#### 选项 B: 从 PyPI 安装（简单快速）
```bash
pip install lerobot
```

### 步骤 2: 验证安装

```bash
python -c "import lerobot; print(f'LeRobot version: {lerobot.__version__}')"
```

### 步骤 3: 运行快速入门示例

```bash
# 从项目根目录
python examples/practice/quick_start.py
```

### 步骤 4: 根据你的目标选择路径

#### 🎮 路径 A: 使用仿真环境（推荐新手，无需硬件）

```bash
# 1. 安装仿真环境支持
pip install -e ".[pusht]"

# 2. 运行数据集示例
python examples/dataset/load_lerobot_dataset.py

# 3. 尝试训练示例（可选，需要 GPU 或较长时间）
python examples/training/train_policy.py
```

#### 🤖 路径 B: 使用真实机器人（需要硬件）

如果你有支持的机器人（如 SO-100, SO-101, HopeJR 等）：

```bash
# 1. 录制演示数据
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

#### 📚 路径 C: 学习现有代码（无需安装额外依赖）

```bash
# 1. 查看数据集示例
cat examples/dataset/load_lerobot_dataset.py

# 2. 查看训练示例
cat examples/training/train_policy.py

# 3. 查看教程
ls examples/tutorial/
```

## 🚀 快速开始（最小步骤）

如果你想最快开始：

```bash
# 1. 安装 LeRobot
pip install -e .

# 2. 运行快速入门
python examples/practice/quick_start.py
```

## 📖 学习资源

- 📚 **完整文档**: https://huggingface.co/docs/lerobot
- 💬 **社区支持**: https://discord.gg/s3KuuzsPFb
- 🐛 **问题反馈**: https://github.com/huggingface/lerobot/issues
- 📦 **数据集 Hub**: https://huggingface.co/lerobot

## ⚠️ 常见问题

### 问题 1: 安装失败
```bash
# 确保在正确的虚拟环境中
# 如果使用 conda:
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e .
```

### 问题 2: 缺少依赖
```bash
# 安装系统依赖（Linux）
sudo apt-get install cmake build-essential python3-dev pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev
```

### 问题 3: 无法下载数据集
```bash
# 登录 Hugging Face
huggingface-cli login
```

## ✅ 检查清单

- [ ] 安装 LeRobot
- [ ] 验证安装成功
- [ ] 运行 `quick_start.py`
- [ ] 阅读 `PRACTICE_GUIDE.md`
- [ ] 选择一个学习路径（仿真/真实机器人/代码学习）
- [ ] 开始实践！

