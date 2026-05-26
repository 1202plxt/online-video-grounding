#!/usr/bin/env python3
"""
Charades-STA 数据集加载和验证工具
"""
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse


class CharadesSTADataset:
    """Charades-STA 数据集加载器"""
    
    def __init__(self, data_dir: str):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据集根目录
        """
        self.data_dir = Path(data_dir)
        self.train_annotations = None
        self.test_annotations = None
        self.video_info = None
        
    def _load_charades_sta_txt(self, txt_path):
        """从 .txt 文件加载 Charades-STA 标注"""
        annotations = []
        if not txt_path.exists():
            return annotations
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 解析: VIDEO_ID START END ## SENTENCE
                parts = line.split('##')
                if len(parts) != 2:
                    continue
                
                video_part = parts[0].strip()
                sentence = parts[1].strip()
                
                video_parts = video_part.split()
                if len(video_parts) < 3:
                    continue
                
                video_id = video_parts[0]
                start_time = float(video_parts[1])
                end_time = float(video_parts[2])
                
                annotations.append({
                    "video_id": video_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "sentence": sentence
                })
        return annotations
    
    def load_annotations(self):
        """加载所有标注文件"""
        # 尝试加载 .json 格式
        train_json = self.data_dir / "charades_sta_train.json"
        test_json = self.data_dir / "charades_sta_test.json"
        
        if train_json.exists():
            with open(train_json, 'r') as f:
                self.train_annotations = json.load(f)
            print(f"✓ 加载训练集 (JSON): {len(self.train_annotations)} 样本")
        else:
            # 尝试加载 .txt 格式
            train_txt = self.data_dir / "charades_sta_train.txt"
            self.train_annotations = self._load_charades_sta_txt(train_txt)
            if self.train_annotations:
                print(f"✓ 加载训练集 (TXT): {len(self.train_annotations)} 样本")

        if test_json.exists():
            with open(test_json, 'r') as f:
                self.test_annotations = json.load(f)
            print(f"✓ 加载测试集 (JSON): {len(self.test_annotations)} 样本")
        else:
            # 尝试加载 .txt 格式
            test_txt = self.data_dir / "charades_sta_test.txt"
            self.test_annotations = self._load_charades_sta_txt(test_txt)
            if self.test_annotations:
                print(f"✓ 加载测试集 (TXT): {len(self.test_annotations)} 样本")

        # 加载原始 Charades 标注
        self._load_charades_original()
        
    def _load_charades_original(self):
        """加载原始 Charades 标注文件"""
        charades_dir = self.data_dir / "Charades"
        if not charades_dir.exists():
            return
        
        self.video_info = {}
        
        # 加载训练和测试 CSV
        for split in ['train', 'test']:
            csv_path = charades_dir / f"Charades_v1_{split}.csv"
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        video_id = row['id']
                        self.video_info[video_id] = {
                            'split': split,
                            'length': float(row['length']) if row['length'] else 0.0,
                            'scene': row['scene'],
                            'subject': row['subject'],
                            'quality': int(row['quality']) if row['quality'] else None,
                        }
        print(f"✓ 加载视频元信息: {len(self.video_info)} 个视频")
    
    def get_video_ids(self, split: str = 'test') -> List[str]:
        """获取指定split的所有视频ID"""
        anns = self.train_annotations if split == 'train' else self.test_annotations
        if not anns:
            return []
        return list(set([ann['video_id'] for ann in anns]))
    
    def verify_videos(self, video_dir: Optional[str] = None) -> Dict:
        """
        验证视频文件是否存在
        
        Args:
            video_dir: 视频文件目录，默认使用 data_dir/videos
            
        Returns:
            验证结果字典
        """
        if video_dir is None:
            video_dir = self.data_dir / "videos"
        else:
            video_dir = Path(video_dir)
        
        all_video_ids = self.get_video_ids('train') + self.get_video_ids('test')
        all_video_ids = list(set(all_video_ids))
        
        found_videos = []
        missing_videos = []
        
        for video_id in all_video_ids:
            # 尝试常见的视频文件名格式
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_path = video_dir / f"{video_id}{ext}"
                if video_path.exists():
                    found_videos.append(video_id)
                    break
            else:
                missing_videos.append(video_id)
        
        return {
            'total': len(all_video_ids),
            'found': len(found_videos),
            'missing': len(missing_videos),
            'missing_ids': missing_videos,
            'video_dir': str(video_dir)
        }
    
    def print_statistics(self):
        """打印数据集统计信息"""
        print("\n" + "="*80)
        print("Charades-STA 数据集统计")
        print("="*80)
        
        if self.train_annotations:
            train_videos = self.get_video_ids('train')
            print(f"\n训练集:")
            print(f"  样本数: {len(self.train_annotations)}")
            print(f"  视频数: {len(train_videos)}")
            print(f"  平均每个视频: {len(self.train_annotations)/len(train_videos):.1f} 个查询")
            
            # 统计时长
            durations = [ann['end_time'] - ann['start_time'] for ann in self.train_annotations]
            print(f"  平均时序长度: {sum(durations)/len(durations):.2f} 秒")
        
        if self.test_annotations:
            test_videos = self.get_video_ids('test')
            print(f"\n测试集:")
            print(f"  样本数: {len(self.test_annotations)}")
            print(f"  视频数: {len(test_videos)}")
            print(f"  平均每个视频: {len(self.test_annotations)/len(test_videos):.1f} 个查询")
            
            durations = [ann['end_time'] - ann['start_time'] for ann in self.test_annotations]
            print(f"  平均时序长度: {sum(durations)/len(durations):.2f} 秒")
        
        # 显示一些示例
        if self.test_annotations:
            print(f"\n示例查询 (测试集):")
            for i, ann in enumerate(self.test_annotations[:5]):
                print(f"  {i+1}. [{ann['video_id']}] {ann['start_time']:.1f}-{ann['end_time']:.1f}: {ann['sentence']}")
        
        print("\n" + "="*80)
    
    def export_for_evaluation(self, output_file: str, split: str = 'test'):
        """
        导出为评估格式
        
        Args:
            output_file: 输出文件路径
            split: 'train' 或 'test'
        """
        anns = self.train_annotations if split == 'train' else self.test_annotations
        if not anns:
            print("✗ 没有找到标注数据")
            return
        
        # 按视频ID分组
        video_groups = {}
        for ann in anns:
            vid = ann['video_id']
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(ann)
        
        # 导出格式
        output_data = []
        for video_id, group in video_groups.items():
            video_data = {
                'video_id': video_id,
                'duration': self.video_info.get(video_id, {}).get('length', 0),
                'annotations': []
            }
            
            for ann in group:
                video_data['annotations'].append({
                    'query': ann['sentence'],
                    'timestamp': [ann['start_time'], ann['end_time']]
                })
            
            output_data.append(video_data)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ 已导出 {len(output_data)} 个视频到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Charades-STA 数据集验证工具')
    parser.add_argument('--data_dir', type=str, default='./data/CharadesSTA',
                       help='数据集目录')
    parser.add_argument('--video_dir', type=str, default=None,
                       help='视频文件目录（可选）')
    parser.add_argument('--verify', action='store_true',
                       help='验证视频文件')
    parser.add_argument('--export', type=str, default=None,
                       help='导出评估数据到指定文件')
    parser.add_argument('--export_split', type=str, default='test',
                       choices=['train', 'test'],
                       help='导出哪个split的数据')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Charades-STA 数据集工具")
    print("="*80)
    
    # 加载数据集
    dataset = CharadesSTADataset(args.data_dir)
    dataset.load_annotations()
    
    # 打印统计信息
    dataset.print_statistics()
    
    # 验证视频
    if args.verify:
        print("\n验证视频文件...")
        result = dataset.verify_videos(args.video_dir)
        print(f"  视频目录: {result['video_dir']}")
        print(f"  总计: {result['total']}")
        print(f"  找到: {result['found']}")
        print(f"  缺失: {result['missing']}")
        
        if result['missing'] > 0:
            print(f"  缺失的视频ID: {result['missing_ids'][:10]}")
            if len(result['missing_ids']) > 10:
                print(f"  ... 还有 {len(result['missing_ids']) - 10} 个")
    
    # 导出数据
    if args.export:
        dataset.export_for_evaluation(args.export, args.export_split)


if __name__ == "__main__":
    main()
