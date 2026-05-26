"""
示例：如何使用评估函数与UniTime进行对比
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation_metrics import evaluate_standard_vtg, print_comparison_with_Unitime

def generate_dummy_data():
    """生成模拟的预测和GT数据（仅作演示）"""
    # 模拟的GT段
    gt_segments_list = [
        {"start": 5.0, "end": 8.0},  # 样本1的GT
        {"start": 12.0, "end": 15.0},  # 样本2的GT
        {"start": 20.0, "end": 25.0},  # 样本3的GT
    ]
    
    # 模拟的预测段（每个样本有按置信度排序的多个预测）
    pred_segments_list = [
        # 样本1的预测
        [
            {"start": 5.2, "end": 7.8, "confidence": 0.95},
            {"start": 4.5, "end": 8.5, "confidence": 0.85},
            {"start": 6.0, "end": 7.0, "confidence": 0.75},
        ],
        # 样本2的预测
        [
            {"start": 11.8, "end": 15.2, "confidence": 0.90},
            {"start": 10.0, "end": 16.0, "confidence": 0.80},
        ],
        # 样本3的预测
        [
            {"start": 21.0, "end": 24.0, "confidence": 0.88},
            {"start": 19.0, "end": 26.0, "confidence": 0.78},
            {"start": 22.0, "end": 23.0, "confidence": 0.68},
        ],
    ]
    
    return pred_segments_list, gt_segments_list

def main():
    print("视频时序定位评估示例")
    print("="*60)
    
    # 1. 生成模拟数据（实际使用时替换为您的真实数据）
    pred_segments_list, gt_segments_list = generate_dummy_data()
    
    print(f"样本数量: {len(pred_segments_list)}")
    
    # 2. 运行标准评估
    print("\n运行标准VTG评估...")
    metrics = evaluate_standard_vtg(
        pred_segments_list, 
        gt_segments_list,
        tiou_thresholds=[0.3, 0.5, 0.7],
        top_ks=[1, 5]
    )
    
    print("\n标准评估指标:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.2f}")
    
    # 3. 打印与UniTime的对比
    print_comparison_with_Unitime(metrics)
    
    print("\n提示：")
    print("  - 要使用真实数据，请替换 generate_dummy_data() 为您的数据加载代码")
    print("  - 确保Charades-STA数据集使用标准的train/test划分")
    print("  - UniTime的数值来自论文：Universal Video Temporal Grounding with Generative Multi-modal Large Language Models")

if __name__ == "__main__":
    main()
