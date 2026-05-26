#!/usr/bin/env python3
"""
Charades-STA 数据集下载脚本
下载并准备视频时序定位任务所需的完整数据集
"""
import os
import sys
import urllib.request
import zipfile
import tarfile
import argparse
from pathlib import Path

# 可选的 tqdm 依赖
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，使用简单的进度显示
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', None)
            self.desc = kwargs.get('desc', '')
            if self.desc:
                print(f"{self.desc}...")
        
        def update(self, n):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass

# 官方下载链接
DOWNLOAD_URLS = {
    # Charades 完整数据集（包含视频）
    "charades_videos": "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1.zip",
    # Charades 标注文件
    "charades_annotations": "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades.zip",
    # Charades-STA 标注（来自DRN官方仓库）
    "charades_sta_train": "https://raw.githubusercontent.com/Alvin-Zeng/DRN/master/data/CharadesSTA/Charades_sta_train.txt",
    "charades_sta_test": "https://raw.githubusercontent.com/Alvin-Zeng/DRN/master/data/CharadesSTA/Charades_sta_test.txt",
}

class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, dest_path, description="Downloading"):
    """下载文件并显示进度条"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{description}")
    print(f"  From: {url}")
    print(f"  To:   {dest_path}")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                               desc=dest_path.name) as t:
            urllib.request.urlretrieve(url, filename=dest_path, 
                                    reporthook=t.update_to)
        print(f"  ✓ Download complete!")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    print(f"\nExtracting {zip_path.name}")
    print(f"  To: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件总数用于进度条
            file_list = zip_ref.infolist()
            for file in tqdm(file_list, desc="Extracting"):
                zip_ref.extract(file, extract_to)
        print(f"  ✓ Extraction complete!")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False

def extract_tar(tar_path, extract_to):
    """解压TAR文件（如果需要）"""
    tar_path = Path(tar_path)
    extract_to = Path(extract_to)
    
    print(f"\nExtracting {tar_path.name}")
    print(f"  To: {extract_to}")
    
    try:
        with tarfile.open(tar_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"  ✓ Extraction complete!")
        return True
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        return False

def parse_charades_sta_line(line):
    """解析Charades-STA的一行标注"""
    line = line.strip()
    if not line:
        return None
    
    # 格式: VIDEO_ID START END ## SENTENCE
    parts = line.split('##')
    if len(parts) != 2:
        return None
    
    video_part = parts[0].strip()
    sentence = parts[1].strip()
    
    video_parts = video_part.split()
    if len(video_parts) < 3:
        return None
    
    video_id = video_parts[0]
    start_time = float(video_parts[1])
    end_time = float(video_parts[2])
    
    return {
        "video_id": video_id,
        "start_time": start_time,
        "end_time": end_time,
        "sentence": sentence
    }

def load_charades_sta_annotations(txt_path):
    """加载Charades-STA标注文件"""
    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            ann = parse_charades_sta_line(line)
            if ann:
                annotations.append(ann)
    return annotations

def main():
    parser = argparse.ArgumentParser(description='Download and prepare Charades-STA dataset')
    parser.add_argument('--data_dir', type=str, default='./data/CharadesSTA',
                       help='Directory to store the dataset')
    parser.add_argument('--skip_videos', action='store_true',
                       help='Skip downloading large video files')
    parser.add_argument('--skip_annotations', action='store_true',
                       help='Skip downloading annotation files')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Charades-STA 数据集下载器")
    print("="*80)
    print(f"目标目录: {data_dir.absolute()}")
    print()
    
    # Step 1: 下载Charades-STA标注文件
    if not args.skip_annotations:
        print("\n" + "="*80)
        print("步骤 1/3: 下载 Charades-STA 标注文件")
        print("="*80)
        
        sta_train_path = data_dir / "Charades_sta_train.txt"
        sta_test_path = data_dir / "Charades_sta_test.txt"
        
        download_file(DOWNLOAD_URLS["charades_sta_train"], sta_train_path,
                     "下载训练集标注")
        download_file(DOWNLOAD_URLS["charades_sta_test"], sta_test_path,
                     "下载测试集标注")
        
        # 验证标注文件
        print("\n验证标注文件...")
        train_ann = load_charades_sta_annotations(sta_train_path)
        test_ann = load_charades_sta_annotations(sta_test_path)
        print(f"  训练集: {len(train_ann)} 个样本")
        print(f"  测试集:  {len(test_ann)} 个样本")
        
        # 保存为JSON格式方便使用
        import json
        with open(data_dir / "charades_sta_train.json", 'w') as f:
            json.dump(train_ann, f, indent=2)
        with open(data_dir / "charades_sta_test.json", 'w') as f:
            json.dump(test_ann, f, indent=2)
        print("  ✓ 已保存为JSON格式")
    
    # Step 2: 下载Charades原始标注
    if not args.skip_annotations:
        print("\n" + "="*80)
        print("步骤 2/3: 下载 Charades 原始标注")
        print("="*80)
        
        annotations_zip = data_dir / "Charades_annotations.zip"
        if download_file(DOWNLOAD_URLS["charades_annotations"], annotations_zip,
                        "下载原始标注ZIP"):
            extract_zip(annotations_zip, data_dir)
    
    # Step 3: 下载视频文件
    if not args.skip_videos:
        print("\n" + "="*80)
        print("步骤 3/3: 下载 Charades 视频文件")
        print("="*80)
        print("⚠️  警告: 视频文件很大 (约 30GB+)，下载需要较长时间！")
        
        videos_zip = data_dir / "Charades_videos.zip"
        if download_file(DOWNLOAD_URLS["charades_videos"], videos_zip,
                        "下载视频ZIP"):
            extract_zip(videos_zip, data_dir / "videos")
    
    print("\n" + "="*80)
    print("下载完成！")
    print("="*80)
    print(f"\n数据集结构:")
    print(f"  {data_dir}/")
    print(f"  ├── Charades_sta_train.txt      (12,408 训练样本)")
    print(f"  ├── Charades_sta_test.txt       (3,720 测试样本)")
    print(f"  ├── charades_sta_train.json     (JSON格式)")
    print(f"  ├── charades_sta_test.json      (JSON格式)")
    if not args.skip_annotations:
        print(f"  ├── Charades/                   (原始标注)")
    if not args.skip_videos:
        print(f"  └── videos/                     (视频文件)")
    
    print("\n提示: 如果视频文件太大，可以使用 --skip_videos 参数跳过")
    print("      然后只下载需要的视频文件")

if __name__ == "__main__":
    main()
