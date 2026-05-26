# Charades-STA 数据集准备指南

本指南帮助您下载、准备和使用 Charades-STA 数据集进行视频时序定位任务。

## 📊 数据集简介

Charades-STA 是基于 Charades 数据集构建的视频时序定位基准数据集：
- **训练集**: 12,408 个文本-时刻对
- **测试集**: 3,720 个文本-时刻对
- **视频总数**: 约 1,863 个视频
- **平均视频时长**: ~30 秒
- **文本查询**: 自然语言描述

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tqdm
```

### 2. 下载标注文件（推荐先试这个）

```bash
cd /workspace/video_temporal_grounding/video_temporal_grounding

# 只下载标注文件（不下载大视频文件）
python download_charades_sta.py --skip_videos
```

### 3. 验证数据集

```bash
python charades_sta_dataset.py
```

### 4. 下载完整数据集（包括视频）

⚠️ **注意**: 视频文件很大（约 30GB+），下载需要较长时间！

```bash
python download_charades_sta.py
```

如果您已经有 Charades 视频文件，可以使用 `--skip_videos` 参数，然后手动指定视频目录：

```bash
# 先下载标注
python download_charades_sta.py --skip_videos

# 然后验证视频文件
python charades_sta_dataset.py --verify --video_dir /path/to/your/videos
```

## 📁 数据集结构

下载完成后，您的目录结构应该是：

```
data/CharadesSTA/
├── Charades_sta_train.txt      # 训练集标注 (12,408 samples)
├── Charades_sta_test.txt       # 测试集标注 (3,720 samples)
├── charades_sta_train.json     # JSON格式（方便Python使用）
├── charades_sta_test.json      # JSON格式
├── Charades/                   # 原始Charades标注
│   ├── Charades_v1_train.csv
│   ├── Charades_v1_test.csv
│   └── ...
└── videos/                     # 视频文件（可选）
    ├── 001YG.mp4
    ├── 003WS.mp4
    └── ...
```

## 📝 标注格式

### Charades-STA 原始格式（TXT）

```
VIDEO_ID START_TIME END_TIME ## SENTENCE
```

示例：
```
3MSZA 24.3 30.4##person turn a light on
3MSZA 24.3 30.4##person flipped the light switch near the door
```

### JSON 格式（我们提供的）

```json
[
  {
    "video_id": "3MSZA",
    "start_time": 24.3,
    "end_time": 30.4,
    "sentence": "person turn a light on"
  },
  ...
]
```

## 🎯 与 UniTime 对比

我们已经在 `evaluation_metrics.py` 中集成了与 UniTime 论文结果的对比功能。

### UniTime 论文结果（Charades-STA）

| 方法 | R@1, IoU=0.3 | R@1, IoU=0.5 | R@1, IoU=0.7 | mIoU |
|------|-------------|-------------|-------------|------|
| Qwen-VL (zero-shot) | 48.7 | 31.5 | 14.2 | 28.9 |
| UniTime (zero-shot) | 62.4 | 45.1 | 23.5 | 38.2 |
| UniTime (finetuned) | 75.8 | 61.2 | 42.1 | 52.6 |

### 使用示例

```python
from evaluation_metrics import evaluate_standard_vtg, print_comparison_with_Unitime
from charades_sta_dataset import CharadesSTADataset

# 加载数据集
dataset = CharadesSTADataset('./data/CharadesSTA')
dataset.load_annotations()

# 准备您的预测结果
# predictions = [{'video_id': ..., 'query': ..., 'predicted_timestamp': [start, end]}, ...]

# 运行评估
metrics = evaluate_standard_vtg(your_predictions, ground_truths)

# 打印对比
print_comparison_with_Unitime(metrics)
```

## 🔗 相关资源

- **Charades 官网**: https://prior.allenai.org/projects/charades
- **Charades-STA 论文**: TALL: Temporal Activity Localization via Language Query (Gao et al., ICCV 2017)
- **UniTime 论文**: Universal Video Temporal Grounding with Generative Multi-modal Large Language Models (NeurIPS 2025)
- **DRN 仓库**: https://github.com/Alvin-Zeng/DRN

## 📋 常见问题

### Q: 视频文件太大了，有没有替代方案？

A: 您可以：
1. 只下载标注文件进行开发
2. 使用预提取的视频特征（如 C3D, SlowFast 等）
3. 先用小部分视频测试

### Q: 如何只下载测试集需要的视频？

A: 您可以使用 `charades_sta_dataset.py` 获取需要的视频ID列表，然后单独下载这些视频。

### Q: 有没有预提取的特征？

A: 是的，很多方法提供预提取的 C3D 或 SlowFast 特征，您可以参考 DRN 或其他项目的仓库。

## 📄 引用

如果您使用 Charades-STA 数据集，请引用：

```bibtex
@inproceedings{gao2017tall,
  title={Tall: Temporal activity localization via language query},
  author={Gao, Jiyang and Sun, Chen and Yang, Zhenheng and Nevatia, Ram},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5267--5275},
  year={2017}
}

@inproceedings{sigurdsson2016hollywood,
  title={Hollywood in homes: Crowdsourcing data collection for activity understanding},
  author={Sigurdsson, Gunnar A and Varol, G{\"u}l and Wang, Xiaolong and Farhadi, Ali and Laptev, Ivan and Gupta, Abhinav},
  booktitle={European conference on computer vision},
  pages={510--526},
  year={2016},
  organization={Springer}
}
```
