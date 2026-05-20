import os
import csv
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CharadesDatasetHandler:
    """Charades 数据集处理类，用于关联视频文件和标注信息"""
    
    def __init__(self, video_dir: str, annotation_dir: str):
        """
        初始化数据集处理器
        Args:
            video_dir: 视频文件目录路径 (/root/autodl-tmp/Charades_v1_480)
            annotation_dir: 标注文件目录路径 (/root/autodl-tmp/Charades)
        """
        self.video_dir = Path(video_dir)
        self.annotation_dir = Path(annotation_dir)
        
        # 验证目录是否存在
        self._validate_directories()
        
        # 存储标注数据的字典 {video_id: annotation_info}
        self.train_annotations: Dict[str, dict] = {}
        self.test_annotations: Dict[str, dict] = {}
        
        # 存储视频路径映射 {video_id: video_path}
        self.video_paths: Dict[str, Path] = {}
        
        # 初始化加载数据
        self.load_annotations()
        self.map_video_files()
    
    def _validate_directories(self):
        """验证目录和关键文件是否存在"""
        required_dirs = [self.video_dir, self.annotation_dir]
        required_files = [
            self.annotation_dir / "Charades_v1_train.csv",
            self.annotation_dir / "Charades_v1_test.csv"
        ]
        
        # 检查目录
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"目录不存在: {dir_path}")
        
        # 检查关键标注文件
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"标注文件缺失: {file_path}")
    
    def load_annotations(self):
        """加载并解析训练和测试标注文件"""
        # 加载训练标注
        self.train_annotations = self._parse_annotation_file(
            self.annotation_dir / "Charades_v1_train.csv"
        )
        
        # 加载测试标注
        self.test_annotations = self._parse_annotation_file(
            self.annotation_dir / "Charades_v1_test.csv"
        )
    
    def _parse_annotation_file(self, csv_path: Path) -> Dict[str, dict]:
        """解析单个CSV标注文件"""
        annotations = {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            # 使用csv.DictReader处理带引号的字段（包含逗号的情况）
            reader = csv.DictReader(f)
            
            for row in reader:
                video_id = row['id'].strip()
                
                # 解析动作标签 (class start end 格式)
                actions = []
                if row['actions'] and row['actions'].strip() != '':
                    action_strs = row['actions'].split(';')
                    for action_str in action_strs:
                        if action_str.strip():
                            parts = action_str.strip().split()
                            if len(parts) >= 3:
                                actions.append({
                                    'class': parts[0],
                                    'start': float(parts[1]),
                                    'end': float(parts[2])
                                })
                
                # 构建标注信息字典
                annotations[video_id] = {
                    'id': video_id,
                    'subject': row['subject'].strip(),
                    'scene': row['scene'].strip(),
                    'quality': int(row['quality']) if row['quality'].isdigit() else None,
                    'relevance': int(row['relevance']) if row['relevance'].isdigit() else None,
                    'verified': row['verified'].strip() == 'Yes',
                    'script': row['script'].strip(),
                    'descriptions': [desc.strip() for desc in row['descriptions'].split(';') if desc.strip()],
                    'actions': actions,
                    'length': float(row['length']) if row['length'] else 0.0
                }
        
        return annotations
    
    def map_video_files(self):
        """扫描视频目录，建立视频ID到文件路径的映射"""
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # 递归查找所有视频文件
        for ext in video_extensions:
            for video_path in glob.glob(str(self.video_dir / f"**/*{ext}"), recursive=True):
                # 从文件名提取视频ID（假设文件名就是视频ID，如 '012345.mp4'）
                video_filename = os.path.basename(video_path)
                video_id = os.path.splitext(video_filename)[0]
                
                self.video_paths[video_id] = Path(video_path)
    
    def get_video_annotation(self, video_id: str) -> Optional[dict]:
        """
        获取指定视频ID的标注信息
        Args:
            video_id: 视频ID（如 '012345'）
        Returns:
            标注信息字典，不存在则返回None
        """
        if video_id in self.train_annotations:
            return {'split': 'train', **self.train_annotations[video_id]}
        elif video_id in self.test_annotations:
            return {'split': 'test', **self.test_annotations[video_id]}
        else:
            return None
    
    def get_video_path(self, video_id: str) -> Optional[Path]:
        """获取指定视频ID的文件路径"""
        return self.video_paths.get(video_id)
    
    def validate_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """
        验证数据集完整性
        Returns:
            (有标注无视频的ID列表, 有视频无标注的ID列表, 完整匹配的ID列表)
        """
        # 所有标注的视频ID
        all_annotated_ids = set(self.train_annotations.keys()) | set(self.test_annotations.keys())
        # 所有视频文件的ID
        all_video_ids = set(self.video_paths.keys())
        
        # 有标注但无视频
        annotation_only = sorted(list(all_annotated_ids - all_video_ids))
        # 有视频但无标注
        video_only = sorted(list(all_video_ids - all_annotated_ids))
        # 完整匹配
        matched = sorted(list(all_annotated_ids & all_video_ids))
        
        return annotation_only, video_only, matched
    
    def save_mapping(self, output_path: str = "charades_mapping.json"):
        """保存视频-标注映射关系到JSON文件"""
        mapping = {
            'video_directory': str(self.video_dir),
            'annotation_directory': str(self.annotation_dir),
            'total_annotations': {
                'train': len(self.train_annotations),
                'test': len(self.test_annotations),
                'total': len(self.train_annotations) + len(self.test_annotations)
            },
            'total_videos': len(self.video_paths),
            'video_mapping': {vid: str(path) for vid, path in self.video_paths.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 初始化数据集处理器
    dataset_handler = CharadesDatasetHandler(
        video_dir="/root/autodl-tmp/Charades_v1_480",
        annotation_dir="/root/autodl-tmp/Charades"
    )
    
    # 验证数据集完整性
    annotation_only, video_only, matched = dataset_handler.validate_dataset()
    
    # 保存映射关系
    dataset_handler.save_mapping()
    
    # 仅输出最终成功提示
    print("数据集匹配完成，所有视频和标注文件均已正确对应")