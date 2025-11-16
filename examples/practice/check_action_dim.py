#!/usr/bin/env python3
"""
查看动作特征的维度

用法:
    python check_action_dim.py <dataset_repo_id>
    
示例:
    python check_action_dim.py lerobot/pusht
    python check_action_dim.py lerobot/aloha_mobile_cabinet
"""

import sys
from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

def check_action_dimensions(repo_id: str):
    """检查动作特征的维度"""
    
    print(f"\n{'='*60}")
    print(f"数据集: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        # 方法 1: 从元数据查看
        print("【方法 1: 从元数据查看】")
        ds_meta = LeRobotDatasetMetadata(repo_id)
        
        # 查找动作特征
        action_features = {}
        for key, feature in ds_meta.features.items():
            if key.startswith("action"):
                action_features[key] = feature
        
        if action_features:
            print("动作特征:")
            for key, feature in action_features.items():
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
                    elif len(shape) > 1:
                        print(f"    维度: {shape}")
                        print(f"    总元素数: {eval('*'.join(map(str, shape)))}")
                
                if names:
                    if isinstance(names, list):
                        print(f"    动作名称: {names}")
                        print(f"    维度数: {len(names)}")
                    elif isinstance(names, dict) and "axes" in names:
                        print(f"    动作名称: {names['axes']}")
                        print(f"    维度数: {len(names['axes'])}")
                        print(f"    动作组成:")
                        for i, name in enumerate(names['axes']):
                            print(f"      [{i}] {name}")
        else:
            print("  未找到动作特征")
        
        # 方法 2: 从实际数据查看
        print(f"\n【方法 2: 从实际数据查看】")
        try:
            dataset = LeRobotDataset(repo_id, episodes=[0])  # 只加载第一个回合
            sample = dataset[0]
            
            action_keys = [k for k in sample.keys() if k.startswith("action")]
            
            if action_keys:
                print("动作特征:")
                for key in action_keys:
                    value = sample[key]
                    print(f"\n  {key}:")
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    
                    # 计算维度
                    if len(value.shape) == 1:
                        dim = value.shape[0]
                        print(f"    维度数: {dim}")
                        if dim <= 20:  # 如果维度不多，显示值
                            print(f"    动作值: {value.tolist()}")
                    elif len(value.shape) > 1:
                        print(f"    维度: {value.shape}")
                        print(f"    总元素数: {value.numel()}")
                        # 如果是时间序列动作
                        if len(value.shape) == 2:
                            print(f"    时间步数: {value.shape[0]}")
                            print(f"    动作维度: {value.shape[1]}")
            else:
                print("  未找到动作特征")
        except Exception as e:
            print(f"  无法加载实际数据: {e}")
        
        # 方法 3: 转换为策略特征查看
        print(f"\n【方法 3: 从策略特征查看】")
        try:
            policy_features = dataset_to_policy_features(ds_meta.features)
            
            action_policy_features = {
                k: v for k, v in policy_features.items() 
                if v.type == FeatureType.ACTION
            }
            
            if action_policy_features:
                print("动作特征（策略格式）:")
                for key, feature in action_policy_features.items():
                    print(f"\n  {key}:")
                    print(f"    类型: {feature.type.value}")
                    print(f"    形状: {feature.shape}")
                    
                    # 计算维度
                    if len(feature.shape) == 1:
                        dim = feature.shape[0]
                        print(f"    维度数: {dim}")
                    elif len(feature.shape) > 1:
                        print(f"    维度: {feature.shape}")
                        print(f"    总元素数: {eval('*'.join(map(str, feature.shape)))}")
            else:
                print("  未找到动作特征")
        except Exception as e:
            print(f"  转换失败: {e}")
        
        # 总结
        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        
        # 找到主要的 action
        if "action" in ds_meta.features:
            main_action = ds_meta.features["action"]
            shape = main_action.get("shape", [])
            names = main_action.get("names", None)
            
            if isinstance(shape, (list, tuple)) and len(shape) == 1:
                dim = shape[0]
                print(f"主要动作特征 'action' 的维度: {dim}")
                
                if names:
                    if isinstance(names, dict) and "axes" in names:
                        print(f"动作组成:")
                        for i, name in enumerate(names['axes']):
                            print(f"  [{i}] {name}")
            else:
                print(f"主要动作特征 'action' 的形状: {shape}")
        else:
            print("未找到 'action' 特征")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_action_dim.py <dataset_repo_id>")
        print("\n示例:")
        print("  python check_action_dim.py lerobot/pusht")
        print("  python check_action_dim.py lerobot/aloha_mobile_cabinet")
        print("\n可用数据集:")
        print("  - lerobot/pusht")
        print("  - lerobot/aloha_mobile_cabinet")
        print("  - lerobot/aloha_sweep_with_hook")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    check_action_dimensions(repo_id)

