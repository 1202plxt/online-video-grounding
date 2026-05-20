import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import os

# 1. 字体配置 (请确保路径正确)
font_path = '/root/autodl-tmp/video_temporal_grounding/simsunb.ttf'
my_font = FontProperties(fname=font_path)

def plot_line_chart_cn():
    # 指标翻译
    metrics = ['平均窗口F1', '总体窗口F1', '平均事件tIoU', 'tIoU@0.3', 'tIoU@0.5', '精确率', '召回率']
    
    # 数据
    hmvtg_5 = [0.6709, 0.7231, 0.4816, 0.7912, 0.6705, 0.6010, 0.7581]
    baseline_5 = [0.6190, 0.6462, 0.3895, 0.6723, 0.5229, 0.4156, 0.7051]
    hmvtg_3 = [0.7057, 0.7713, 0.5545, 0.8631, 0.7463, 0.6811, 0.8210]
    baseline_3 = [0.6903, 0.7674, 0.5158, 0.8115, 0.6984, 0.6520, 0.7845]

    plt.figure(figsize=(12, 7))
    
    plt.plot(metrics, hmvtg_5, marker='s', markersize=8, linewidth=2, label='HMVTG (5并并行)', color='#d73027')
    plt.plot(metrics, baseline_5, marker='o', markersize=8, linewidth=2, linestyle='--', label='Baseline (5并行)', color='#7b9ce6')
    plt.plot(metrics, hmvtg_3, marker='D', markersize=8, linewidth=2, label='HMVTG (3并行)', color='#1a9850')
    plt.plot(metrics, baseline_3, marker='^', markersize=8, linewidth=2, linestyle='--', label='Baseline (3并行)', color='#a6d96a')

    # 应用中文字体
    plt.title('性能对比：HMVTG与基线模型在7项指标下的表现', fontproperties=my_font, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('分数 (Score)', fontproperties=my_font, fontsize=12)
    plt.xlabel('评估指标', fontproperties=my_font, fontsize=12)
    plt.xticks(range(len(metrics)), metrics, fontproperties=my_font, fontsize=10)
    
    plt.ylim(0.38, 0.90)
    plt.yticks(np.arange(0.38, 0.92, 0.04))
    plt.grid(True, which='both', linestyle=':', alpha=0.6, color='gray')
    plt.legend(prop=my_font, loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    save_path = '/root/autodl-tmp/results/Figure_4_1_a_Line_CN.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 折线图已保存至: {save_path}")

if __name__ == '__main__':
    plot_line_chart_cn()