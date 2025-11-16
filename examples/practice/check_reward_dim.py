#!/usr/bin/env python3
"""
查看奖励特征的维度

用法:
    python check_reward_dim.py <dataset_repo_id>
    
示例:
    python check_reward_dim.py lerobot/pusht
    python check_reward_dim.py lerobot/aloha_mobile_cabinet
"""

import sys
from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

def check_reward_dimensions(repo_id: str):
    """检查奖励特征的维度"""
    
    print(f"\n{'='*60}")
    print(f"数据集: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        # 方法 1: 从元数据查看
        print("【方法 1: 从元数据查看】")
        ds_meta = LeRobotDatasetMetadata(repo_id)
        
        # 查找奖励特征
        reward_features = {}
        for key, feature in ds_meta.features.items():
            if "reward" in key.lower():
                reward_features[key] = feature
        
        if reward_features:
            print("奖励特征:")
            for key, feature in reward_features.items():
                shape = feature.get("shape", "N/A")
                dtype = feature.get("dtype", "N/A")
                names = feature.get("names", None)
                
                print(f"\n  {key}:")
                print(f"    形状: {shape}")
                print(f"    数据类型: {dtype}")
                
                # 计算维度
                if isinstance(shape, (list, tuple)):
                    if len(shape) == 1:
                        dim = shape[0]
                        print(f"    维度数: {dim}")
                        if dim == 1:
                            print(f"    类型: 标量奖励值")
                    elif len(shape) > 1:
                        print(f"    维度: {shape}")
                        print(f"    总元素数: {eval('*'.join(map(str, shape)))}")
                
                if names:
                    print(f"    名称: {names}")
        else:
            print("  未找到奖励特征")
            print("  注意: 某些数据集（如模仿学习数据集）可能不包含奖励信息")
        
        # 方法 2: 从实际数据查看
        print(f"\n【方法 2: 从实际数据查看】")
        try:
            dataset = LeRobotDataset(repo_id, episodes=[0])  # 只加载第一个回合
            sample = dataset[0]
            
            reward_keys = [k for k in sample.keys() if "reward" in k.lower()]
            
            if reward_keys:
                print("奖励特征:")
                for key in reward_keys:
                    value = sample[key]
                    print(f"\n  {key}:")
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    
                    # 计算维度
                    if len(value.shape) == 0:
                        # 标量
                        print(f"    类型: 标量")
                        print(f"    值: {value.item()}")
                    elif len(value.shape) == 1:
                        dim = value.shape[0]
                        print(f"    维度数: {dim}")
                        if dim == 1:
                            print(f"    类型: 标量奖励值（包装在数组中）")
                            print(f"    值: {value.item()}")
                        else:
                            print(f"    值: {value.tolist()}")
                    else:
                        print(f"    维度: {value.shape}")
                        print(f"    总元素数: {value.numel()}")
                
                # 统计奖励值
                print(f"\n  奖励值统计（前10个样本）:")
                try:
                    for i in range(min(10, len(dataset))):
                        sample = dataset[i]
                        if reward_keys[0] in sample:
                            reward = sample[reward_keys[0]]
                            if isinstance(reward, (int, float)) or (hasattr(reward, 'numel') and reward.numel() == 1):
                                reward_val = reward.item() if hasattr(reward, 'item') else reward
                                print(f"    样本 {i}: {reward_val}")
                except Exception as e:
                    print(f"    无法统计: {e}")
            else:
                print("  未找到奖励特征")
                print("  注意: 某些数据集（如模仿学习数据集）可能不包含奖励信息")
        except Exception as e:
            print(f"  无法加载实际数据: {e}")
        
        # 方法 3: 转换为策略特征查看
        print(f"\n【方法 3: 从策略特征查看】")
        try:
            policy_features = dataset_to_policy_features(ds_meta.features)
            
            reward_policy_features = {
                k: v for k, v in policy_features.items() 
                if v.type == FeatureType.REWARD
            }
            
            if reward_policy_features:
                print("奖励特征（策略格式）:")
                for key, feature in reward_policy_features.items():
                    print(f"\n  {key}:")
                    print(f"    类型: {feature.type.value}")
                    print(f"    形状: {feature.shape}")
                    
                    # 计算维度
                    if len(feature.shape) == 1:
                        dim = feature.shape[0]
                        print(f"    维度数: {dim}")
                        if dim == 1:
                            print(f"    类型: 标量奖励值")
            else:
                print("  未找到奖励特征")
                print("  注意: 某些数据集（如模仿学习数据集）可能不包含奖励信息")
        except Exception as e:
            print(f"  转换失败: {e}")
        
        # 总结
        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        
        # 找到主要的 reward
        if "reward" in ds_meta.features:
            main_reward = ds_meta.features["reward"]
            shape = main_reward.get("shape", [])
            dtype = main_reward.get("dtype", "N/A")
            
            print(f"主要奖励特征 'reward':")
            print(f"  形状: {shape}")
            print(f"  数据类型: {dtype}")
            
            if isinstance(shape, (list, tuple)) and len(shape) == 1 and shape[0] == 1:
                print(f"  类型: 标量奖励值（每个时间步一个奖励值）")
                print(f"  用途: 用于强化学习训练，评估动作的好坏")
        elif "next.reward" in ds_meta.features:
            print(f"找到 'next.reward' 特征（下一个状态的奖励）")
        else:
            print("未找到奖励特征")
            print("注意: 此数据集可能不包含奖励信息，可能是模仿学习数据集")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_reward_dim.py <dataset_repo_id>")
        print("\n示例:")
        print("  python check_reward_dim.py lerobot/pusht")
        print("  python check_reward_dim.py lerobot/aloha_mobile_cabinet")
        print("\n注意:")
        print("  - 某些数据集（如模仿学习数据集）可能不包含奖励信息")
        print("  - 奖励特征通常用于强化学习训练")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    check_reward_dimensions(repo_id)

