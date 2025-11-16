#!/usr/bin/env python3
"""
查看视觉特征的维度

用法:
    python check_visual_dim.py <dataset_repo_id>
    
示例:
    python check_visual_dim.py lerobot/pusht
    python check_visual_dim.py lerobot/aloha_mobile_cabinet
"""

import sys
from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features

def check_visual_dimensions(repo_id: str):
    """检查视觉特征的维度"""
    
    print(f"\n{'='*60}")
    print(f"数据集: {repo_id}")
    print(f"{'='*60}\n")
    
    try:
        # 方法 1: 从元数据查看
        print("【方法 1: 从元数据查看】")
        ds_meta = LeRobotDatasetMetadata(repo_id)
        
        # 查找视觉特征（图像/视频）
        visual_features = {}
        for key, feature in ds_meta.features.items():
            if feature.get("dtype") in ["image", "video"]:
                visual_features[key] = feature
        
        if visual_features:
            print("视觉特征:")
            for key, feature in visual_features.items():
                shape = feature.get("shape", "N/A")
                dtype = feature.get("dtype", "N/A")
                names = feature.get("names", None)
                
                print(f"\n  {key}:")
                print(f"    数据类型: {dtype}")
                print(f"    形状: {shape}")
                
                # 解析维度
                if isinstance(shape, (list, tuple)) and len(shape) == 3:
                    h, w, c = shape
                    print(f"    高度 (Height): {h}")
                    print(f"    宽度 (Width): {w}")
                    print(f"    通道数 (Channels): {c}")
                    print(f"    总像素数: {h * w * c}")
                    print(f"    格式: (H, W, C) - 数据集存储格式")
                    
                    if names:
                        if isinstance(names, list) and len(names) == 3:
                            print(f"    维度名称: {names}")
                else:
                    print(f"    形状: {shape}")
        else:
            print("  未找到视觉特征")
        
        # 方法 2: 从实际数据查看
        print(f"\n【方法 2: 从实际数据查看】")
        try:
            dataset = LeRobotDataset(repo_id, episodes=[0])  # 只加载第一个回合
            sample = dataset[0]
            
            # 查找所有图像键
            image_keys = [k for k in sample.keys() if "image" in k.lower() or "video" in k.lower()]
            
            if image_keys:
                print("视觉特征:")
                for key in image_keys:
                    value = sample[key]
                    print(f"\n  {key}:")
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    
                    # 解析维度
                    if len(value.shape) == 3:
                        # 可能是 (C, H, W) 或 (H, W, C)
                        if value.shape[0] == 3 or value.shape[0] == 1:
                            # 可能是 (C, H, W) - PyTorch 格式
                            c, h, w = value.shape
                            print(f"    格式: (C, H, W) - PyTorch channel-first 格式")
                            print(f"    通道数 (Channels): {c}")
                            print(f"    高度 (Height): {h}")
                            print(f"    宽度 (Width): {w}")
                        elif value.shape[2] == 3 or value.shape[2] == 1:
                            # 可能是 (H, W, C) - 数据集格式
                            h, w, c = value.shape
                            print(f"    格式: (H, W, C) - 数据集存储格式")
                            print(f"    高度 (Height): {h}")
                            print(f"    宽度 (Width): {w}")
                            print(f"    通道数 (Channels): {c}")
                        else:
                            print(f"    维度: {value.shape}")
                    elif len(value.shape) == 4:
                        # 可能是批次格式 (B, C, H, W) 或 (B, H, W, C)
                        print(f"    批次格式: {value.shape}")
                        if value.shape[1] == 3 or value.shape[1] == 1:
                            b, c, h, w = value.shape
                            print(f"    格式: (B, C, H, W)")
                            print(f"    批次大小: {b}")
                            print(f"    通道数: {c}")
                            print(f"    高度: {h}")
                            print(f"    宽度: {w}")
                        elif value.shape[3] == 3 or value.shape[3] == 1:
                            b, h, w, c = value.shape
                            print(f"    格式: (B, H, W, C)")
                            print(f"    批次大小: {b}")
                            print(f"    高度: {h}")
                            print(f"    宽度: {w}")
                            print(f"    通道数: {c}")
                    else:
                        print(f"    维度: {value.shape}")
                    
                    print(f"    总元素数: {value.numel()}")
            else:
                print("  未找到视觉特征")
        except Exception as e:
            print(f"  无法加载实际数据: {e}")
        
        # 方法 3: 转换为策略特征查看
        print(f"\n【方法 3: 从策略特征查看】")
        try:
            policy_features = dataset_to_policy_features(ds_meta.features)
            
            visual_policy_features = {
                k: v for k, v in policy_features.items() 
                if v.type == FeatureType.VISUAL
            }
            
            if visual_policy_features:
                print("视觉特征（策略格式 - channel-first）:")
                for key, feature in visual_policy_features.items():
                    print(f"\n  {key}:")
                    print(f"    类型: {feature.type.value}")
                    print(f"    形状: {feature.shape}")
                    
                    # 解析维度
                    if len(feature.shape) == 3:
                        c, h, w = feature.shape
                        print(f"    格式: (C, H, W) - PyTorch channel-first 格式")
                        print(f"    通道数 (Channels): {c}")
                        print(f"    高度 (Height): {h}")
                        print(f"    宽度 (Width): {w}")
                        print(f"    总像素数: {c * h * w}")
            else:
                print("  未找到视觉特征")
        except Exception as e:
            print(f"  转换失败: {e}")
        
        # 总结
        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        
        # 找到所有视觉特征
        visual_keys = [k for k in ds_meta.features.keys() 
                      if ds_meta.features[k].get("dtype") in ["image", "video"]]
        
        if visual_keys:
            print(f"找到 {len(visual_keys)} 个视觉特征:")
            for key in visual_keys:
                feature = ds_meta.features[key]
                shape = feature.get("shape", [])
                if isinstance(shape, (list, tuple)) and len(shape) == 3:
                    h, w, c = shape
                    print(f"  - {key}: {h}x{w}x{c} (H×W×C)")
                else:
                    print(f"  - {key}: {shape}")
        else:
            print("未找到视觉特征")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_visual_dim.py <dataset_repo_id>")
        print("\n示例:")
        print("  python check_visual_dim.py lerobot/pusht")
        print("  python check_visual_dim.py lerobot/aloha_mobile_cabinet")
        print("\n可用数据集:")
        print("  - lerobot/pusht")
        print("  - lerobot/aloha_mobile_cabinet")
        print("  - lerobot/aloha_sweep_with_hook")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    check_visual_dimensions(repo_id)

