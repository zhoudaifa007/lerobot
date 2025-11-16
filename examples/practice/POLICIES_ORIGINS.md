# LeRobot Policies 来源分析

本文档分析 LeRobot 框架中各个策略（Policies）的来源和开发者信息。

## 📋 策略列表

LeRobot 框架支持以下策略：

1. **ACT** - Action Chunking with Transformers
2. **Diffusion Policy** - Diffusion-based Policy Learning
3. **TD-MPC** - Temporal Difference Learning for Model Predictive Control
4. **VQ-BeT** - Vector Quantized Behavior Transformer
5. **π₀ (pi0)** - Physical Intelligence π₀
6. **π₀.₅ (pi05)** - Physical Intelligence π₀.₅
7. **SmolVLA** - Small Vision-Language-Action Model
8. **GR00T** - NVIDIA GR00T Foundation Model
9. **SAC** - Soft Actor-Critic

---

## 🏢 各策略来源详情

### 1. ACT (Action Chunking with Transformers)

**来源机构**：Stanford University / UC Berkeley

**主要作者**：
- Tony Z. Zhao
- Vikash Kumar
- Sergey Levine
- Chelsea Finn

**论文**：
- Title: "Learning fine-grained bimanual manipulation with low-cost hardware"
- arXiv: [2304.13705](https://arxiv.org/abs/2304.13705)
- Year: 2023
- Website: https://tonyzhaozh.github.io/aloha

**特点**：
- 用于精细双手操作任务
- 适用于低成本硬件
- 使用 Transformer 进行动作分块

**代码位置**：`src/lerobot/policies/act/`

---

### 2. Diffusion Policy

**来源机构**：Columbia University

**主要作者**：
- Cheng Chi
- Zhenjia Xu
- Siyuan Feng
- Eric Cousineau
- Yilun Du
- Benjamin Burchfiel
- Russ Tedrake
- Shuran Song

**论文**：
- Title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
- Journal: The International Journal of Robotics Research
- Year: 2024
- Website: https://diffusion-policy.cs.columbia.edu

**特点**：
- 基于扩散模型的策略学习
- 视觉-运动策略
- 适用于复杂操作任务

**代码位置**：`src/lerobot/policies/diffusion/`

---

### 3. TD-MPC (Temporal Difference Learning for Model Predictive Control)

**来源机构**：UC San Diego

**主要作者**：
- Nicklas Hansen
- Xiaolong Wang
- Hao Su

**论文**：
- Title: "Temporal Difference Learning for Model Predictive Control"
- Conference: ICML 2022
- Website: https://www.nicklashansen.com/td-mpc/

**特点**：
- 结合时间差分学习和模型预测控制
- 支持离线世界模型微调（FOWM）
- 适用于连续控制任务

**代码位置**：`src/lerobot/policies/tdmpc/`

---

### 4. VQ-BeT (Vector Quantized Behavior Transformer)

**来源机构**：New York University (NYU)

**主要作者**：
- Seungjae Lee
- Yibin Wang
- Haritheja Etukuru
- H Jin Kim
- Nur Muhammad Mahi Shafiullah
- Lerrel Pinto

**论文**：
- Title: "Behavior generation with latent actions"
- arXiv: [2403.03181](https://arxiv.org/abs/2403.03181)
- Year: 2024
- Website: https://sjlee.cc/vq-bet/

**特点**：
- 使用向量量化（VQ）进行动作离散化
- 基于 Behavior Transformer (BeT)
- 适用于行为生成任务

**代码位置**：`src/lerobot/policies/vqbet/`

---

### 5. π₀ (pi0) - Physical Intelligence π₀

**来源机构**：**Physical Intelligence** 公司

**主要作者**：
- Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky

**论文**：
- Title: "π₀: A Vision-Language-Action Flow Model for General Robot Control"
- arXiv: [2410.24164](https://arxiv.org/abs/2410.24164)
- Year: 2024
- Repository: https://github.com/Physical-Intelligence/openpi

**特点**：
- 首个通用机器人基础模型
- 视觉-语言-动作模型
- 支持多种机器人和任务
- 48 tokens 长度

**代码位置**：`src/lerobot/policies/pi0/`

---

### 6. π₀.₅ (pi05) - Physical Intelligence π₀.₅

**来源机构**：**Physical Intelligence** 公司

**主要作者**：
- Physical Intelligence 团队（包括 Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, Ury Zhilinsky）

**论文**：
- Title: "π₀.₅: a Vision-Language-Action Model with Open-World Generalization"
- arXiv: [2504.16054](https://arxiv.org/abs/2504.16054)
- Year: 2025
- Repository: https://github.com/Physical-Intelligence/openpi

**特点**：
- π₀ 的进化版本
- 开放世界泛化能力
- 使用 AdaRMS 条件
- 200 tokens 长度
- 离散状态输入

**代码位置**：`src/lerobot/policies/pi05/`

---

### 7. SmolVLA (Small Vision-Language-Action Model)

**来源机构**：**Hugging Face**

**主要作者**：
- Mustafa Shukor
- Dana Aubakirova
- Francesco Capuano
- Pepijn Kooijmans
- Steven Palma
- Adil Zouitine
- Michel Aractingi
- Caroline Pascal
- Martino Russi
- Andres Marafioti
- Simon Alibert
- Matthieu Cord
- Thomas Wolf
- Remi Cadene

**论文**：
- Title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"
- arXiv: [2506.01844](https://arxiv.org/abs/2506.01844)
- Year: 2025

**特点**：
- 轻量级视觉-语言-动作模型
- 成本效益高
- 高效机器人控制
- Hugging Face 团队开发

**代码位置**：`src/lerobot/policies/smolvla/`

---

### 8. GR00T (NVIDIA GR00T Foundation Model)

**来源机构**：**NVIDIA**

**主要作者**：
- NVIDIA 团队（包括 Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, Yuke Zhu）

**论文**：
- Title: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"
- arXiv: [2503.14734](https://arxiv.org/abs/2503.14734)
- Year: 2025
- Website: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- Repository: https://github.com/NVIDIA/Isaac-GR00T
- Model: https://huggingface.co/nvidia/GR00T-N1.5-3B

**特点**：
- 通用人形机器人基础模型
- 开放基础模型
- NVIDIA Isaac 平台
- 适用于人形机器人

**代码位置**：`src/lerobot/policies/groot/`

---

### 9. SAC (Soft Actor-Critic)

**来源机构**：经典强化学习算法（非特定机构）

**论文**：
- Title: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
- arXiv: [1801.01290](https://arxiv.org/abs/1801.01290)
- Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine

**特点**：
- 熵正则化的 Actor-Critic 算法
- 稳定的样本高效学习
- 适用于连续控制环境
- 离线策略算法

**代码位置**：`src/lerobot/policies/sac/`

---

## 📊 策略来源汇总表

| 策略名称 | 来源机构 | 类型 | 年份 |
|---------|---------|------|------|
| **ACT** | Stanford / UC Berkeley | 学术研究 | 2023 |
| **Diffusion Policy** | Columbia University | 学术研究 | 2024 |
| **TD-MPC** | UC San Diego | 学术研究 | 2022 |
| **VQ-BeT** | New York University | 学术研究 | 2024 |
| **π₀ (pi0)** | Physical Intelligence | 公司产品 | 2024 |
| **π₀.₅ (pi05)** | Physical Intelligence | 公司产品 | 2025 |
| **SmolVLA** | Hugging Face | 公司产品 | 2025 |
| **GR00T** | NVIDIA | 公司产品 | 2025 |
| **SAC** | 经典算法 | 学术研究 | 2018 |

---

## 🏢 机构分类

### 学术机构

1. **Stanford University / UC Berkeley**
   - ACT

2. **Columbia University**
   - Diffusion Policy

3. **UC San Diego**
   - TD-MPC

4. **New York University (NYU)**
   - VQ-BeT

### 公司/组织

1. **Physical Intelligence**
   - π₀ (pi0)
   - π₀.₅ (pi05)

2. **Hugging Face**
   - SmolVLA

3. **NVIDIA**
   - GR00T

4. **经典算法**
   - SAC

---

## 🔍 策略特点对比

### 按模型类型分类

| 类型 | 策略 |
|------|------|
| **Transformer-based** | ACT, VQ-BeT, π₀, π₀.₅, SmolVLA, GR00T |
| **Diffusion-based** | Diffusion Policy |
| **Model-based RL** | TD-MPC |
| **Actor-Critic** | SAC |

### 按能力分类

| 能力 | 策略 |
|------|------|
| **视觉-语言-动作** | π₀, π₀.₅, SmolVLA, GR00T |
| **视觉-动作** | ACT, Diffusion Policy, TD-MPC, VQ-BeT |
| **纯强化学习** | SAC |

### 按应用场景分类

| 场景 | 策略 |
|------|------|
| **通用机器人控制** | π₀, π₀.₅, GR00T |
| **精细操作** | ACT, Diffusion Policy |
| **连续控制** | TD-MPC, SAC |
| **行为生成** | VQ-BeT |
| **轻量级部署** | SmolVLA |

---

## 📝 总结

### 机构分布

- **学术机构**：4 个策略（ACT, Diffusion Policy, TD-MPC, VQ-BeT）
- **公司产品**：4 个策略（π₀, π₀.₅, SmolVLA, GR00T）
- **经典算法**：1 个策略（SAC）

### 最新趋势

1. **基础模型**：π₀, π₀.₅, GR00T 都是通用机器人基础模型
2. **视觉-语言-动作**：多个策略支持多模态输入
3. **公司参与**：Physical Intelligence、NVIDIA、Hugging Face 等公司积极参与

### 技术方向

1. **Transformer 架构**：大多数新策略使用 Transformer
2. **多模态学习**：视觉-语言-动作融合
3. **泛化能力**：开放世界泛化成为重点

---

这些策略代表了机器人学习领域的最新进展，从学术研究到工业应用，涵盖了多种技术路线和应用场景。

