#!/usr/bin/env python3
"""
LeRobot 快速入门示例
这个脚本演示了如何使用 LeRobot 加载和查看数据集
"""

import sys
from pathlib import Path

# 检查是否安装了必要的包
try:
    import torch
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from huggingface_hub import HfApi
except ImportError as e:
    print(f"❌ 缺少必要的包: {e}")
    print("\n请先安装 LeRobot:")
    print("  pip install lerobot")
    print("或者从源码安装:")
    print("  pip install -e .")
    sys.exit(1)

def main():
    print("=" * 60)
    print("🤖 LeRobot 快速入门示例")
    print("=" * 60)
    
    # 1. 查看可用的数据集
    print("\n📦 查看可用的数据集...")
    print(f"LeRobot 版本: {lerobot.__version__}")
    print(f"\n可用的数据集数量: {len(lerobot.available_datasets)}")
    print("\n前 10 个数据集:")
    for i, dataset in enumerate(lerobot.available_datasets[:10], 1):
        print(f"  {i}. {dataset}")
    
    # 2. 选择一个小的数据集进行演示（PushT 是一个小的仿真数据集）
    repo_id = "lerobot/pusht"
    print(f"\n📊 加载数据集: {repo_id}")
    
    try:
        # 只加载元数据（不下载完整数据）
        print("  正在获取数据集元数据...")
        ds_meta = LeRobotDatasetMetadata(repo_id)
        
        print(f"\n✅ 数据集信息:")
        print(f"  - 总 episode 数: {ds_meta.total_episodes}")
        print(f"  - 总帧数: {ds_meta.total_frames}")
        print(f"  - 平均每 episode 帧数: {ds_meta.total_frames / ds_meta.total_episodes:.1f}")
        print(f"  - FPS: {ds_meta.fps}")
        print(f"  - 机器人类型: {ds_meta.robot_type}")
        
        if hasattr(ds_meta, 'camera_keys') and ds_meta.camera_keys:
            print(f"  - 相机键: {ds_meta.camera_keys}")
        
        print(f"\n📋 特征列表:")
        for key, feature in list(ds_meta.features.items())[:5]:  # 只显示前5个
            print(f"  - {key}: {feature.get('shape', 'N/A')}")
        
        # 3. 加载第一个 episode 的数据
        print(f"\n📥 加载第一个 episode 的数据...")
        dataset = LeRobotDataset(repo_id, episodes=[0])
        
        print(f"  ✅ 成功加载!")
        print(f"  - 加载的 episode 数: {dataset.num_episodes}")
        print(f"  - 加载的帧数: {dataset.num_frames}")
        
        # 4. 查看第一帧数据
        if dataset.num_frames > 0:
            print(f"\n🔍 查看第一帧数据:")
            first_frame = dataset[0]
            print(f"  数据键: {list(first_frame.keys())}")
            for key, value in first_frame.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        print("\n" + "=" * 60)
        print("✅ 快速入门示例完成!")
        print("=" * 60)
        print("\n💡 下一步:")
        print("  1. 查看 examples/ 目录了解更多示例")
        print("  2. 尝试训练一个策略: python examples/training/train_policy.py")
        print("  3. 查看文档: https://huggingface.co/docs/lerobot")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题（需要从 Hugging Face Hub 下载数据）")
        print("  2. 数据集不存在或已更改")
        print("  3. 需要先登录 Hugging Face: huggingface-cli login")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

