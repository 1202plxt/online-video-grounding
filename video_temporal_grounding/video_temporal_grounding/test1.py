import torch
import json
import warnings
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from dataset import CharadesDatasetHandler
from evaluation_metrics import (
    evaluate_video_complete,
    evaluate_batch_complete,
    merge_contiguous_segments
)

warnings.filterwarnings('ignore')

def get_elapsed_time(start_time):
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = round(elapsed_seconds % 60, 2)
    if hours > 0:
        elapsed_str = f"{hours}小时{minutes}分钟{seconds}秒"
    elif minutes > 0:
        elapsed_str = f"{minutes}分钟{seconds}秒"
    else:
        elapsed_str = f"{seconds}秒"
    return elapsed_str, elapsed_seconds

# ======================== 视频帧采样模块 ========================
def sample_video_frames_sliding_window(video_path, window_size=1, stride=1, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    windows = []
    start_time = 0.0

    while start_time < duration:
        end_time = min(start_time + window_size, duration)
        current_window_frames = []

        time_points = np.linspace(start_time, end_time, num_frames)
        all_frame_indices = [min(int(t * fps), total_frames - 1) for t in time_points]

        for idx in all_frame_indices:
            # 修复 OpenCV 兼容问题
            cap.set(1, idx)
            ret, frame = cap.read()
            if ret:
                current_window_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if current_window_frames:
                    current_window_frames.append(current_window_frames[-1])
                else:
                    current_window_frames.append(np.zeros((360, 640, 3), dtype=np.uint8))

        windows.append({
            "start": start_time,
            "end": end_time,
            "frames": current_window_frames,
            "timestamps": time_points.tolist(),
            "frame_indices": all_frame_indices
        })

        start_time += stride

    cap.release()
    print(f"✅ 视频采样完成 - 总时长: {duration:.2f}秒 | 窗口大小: {window_size}秒 | 步长: {stride}秒 | 采样帧数: {num_frames} | 总窗口数: {len(windows)}")
    return windows, duration

# ======================== 视频定位主模块 ========================
class CharadesVideoGrounding:
    def __init__(self, model_path, video_dir, annotation_dir, device_map="auto",
                 window_size=1, stride=1, num_frames=8, conf_threshold=0.7, max_action_num=5,
                 tiou_thresholds=[0.3, 0.5]):
        print("📥 初始化数据集处理器...")
        self.dataset_handler = CharadesDatasetHandler(video_dir, annotation_dir)

        annotation_only, video_only, matched = self.dataset_handler.validate_dataset()
        print(f"✅ 数据集验证完成 - 匹配视频数: {len(matched)} | 缺失标注: {len(annotation_only)} | 缺失视频: {len(video_only)}")

        print(f"📥 加载Qwen3VL模型...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.window_size = window_size
        self.stride = stride
        self.num_frames = num_frames
        self.conf_threshold = conf_threshold
        self.merge_time_threshold = 0.1
        self.max_action_num = max_action_num
        self.tiou_thresholds = tiou_thresholds

        self.action_classes = self._load_action_classes(annotation_dir)

    def _load_action_classes(self, annotation_dir):
        class_file = Path(annotation_dir) / "Charades_v1_classes.txt"
        action_classes = {}

        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(maxsplit=1)
                        if len(parts) >= 2:
                            action_classes[parts[0]] = parts[1]
        else:
            print(f"⚠️  动作类别文件未找到: {class_file}")

        return action_classes

    def get_video_full_info(self, video_id):
        annotation = self.dataset_handler.get_video_annotation(video_id)
        if not annotation:
            return None

        video_path = self.dataset_handler.get_video_path(video_id)
        if not video_path:
            return None

        return {
            'id': video_id,
            'split': annotation['split'],
            'path': str(video_path),
            'duration': annotation['length'],
            'scene': annotation['scene'],
            'actions': annotation['actions'],
            'script': annotation['script'],
            'descriptions': annotation['descriptions']
        }

    def build_multi_action_prompt(self, window_info, actions_to_detect):
        start_time = window_info["start"]
        end_time = window_info["end"]
        timestamps = window_info["timestamps"]

        frame_desc = [f"Frame {i+1}: {t:.2f}s" for i, t in enumerate(timestamps)]
        time_desc = ", ".join(frame_desc[:3]) + "..."

        action_list = []
        for idx, action in enumerate(actions_to_detect[:self.max_action_num]):
            action_class = action['class']
            action_desc = self.action_classes.get(action_class, f"Unknown action {action_class}")
            action_list.append(f"{idx+1}. {action_class} (time: {action['start']:.1f}-{action['end']:.1f}s)")

        action_list_str = "\n".join(action_list)

        prompt = f"""
You are a fast video action analyzer. Analyze THIS 1-second video segment ({start_time:.1f}-{end_time:.1f}s) and judge ONLY THESE {len(actions_to_detect)} actions:

{action_list_str}

TASK: For each action, output:
1. happens: true/false
2. confidence: 0.0-1.0 (1 decimal place)
3. explanation: MAX 8 WORDS (short reason only)

STRICT OUTPUT FORMAT (ONLY JSON, NO OTHER TEXT):
{{"actions":{{"ACTION_CLASS":{{"happens":true/false,"confidence":0.0,"explanation":"short reason"}}}},"window_info":{{"start":{start_time:.1f},"end":{end_time:.1f}}}}}
"""
        return prompt

    def process_single_video(self, video_id):
        video_info = self.get_video_full_info(video_id)
        if not video_info:
            print(f"❌ 无法找到视频 {video_id} 的完整信息")
            return None

        video_path = video_info['path']
        duration = video_info['duration']
        all_actions = video_info['actions']
        
        # 检查是否有动作标签
        if not all_actions or len(all_actions) == 0:
            print(f"⏭️  跳过视频 {video_id}: 无动作标签")
            return None
            
        actions_to_detect = all_actions[:self.max_action_num]

        if len(all_actions) < self.max_action_num:
            print(f"⚠️  视频 {video_id} 仅包含 {len(all_actions)} 个动作（不足{self.max_action_num}个），将判断全部动作")

        print(f"\n====================================================")
        print(f"📹 开始处理视频: {video_id} (8帧采样+固定前{self.max_action_num}动作+时间段合并)")
        print(f"📍 场景: {video_info['scene']} | 时长: {duration:.2f}秒 | 划分: {video_info['split']}")
        print(f"⚙️  滑窗配置: 窗口={self.window_size}秒 | 步长={self.stride}秒 | 采样帧数={self.num_frames}")
        print(f"📝 待检测动作数: {len(actions_to_detect)} (固定前{self.max_action_num}个)")
        for act in actions_to_detect:
            act_desc = self.action_classes.get(act['class'], '未知动作')
            print(f"   - {act['class']}: {act_desc} (标注时间: {act['start']:.2f}-{act['end']:.2f}秒)")
        print(f"====================================================")

        windows, _ = sample_video_frames_sliding_window(
            video_path,
            window_size=self.window_size,
            stride=self.stride,
            num_frames=self.num_frames
        )

        multi_action_results = {}
        window_raw_results = []
        for action in actions_to_detect:
            multi_action_results[action['class']] = []

        for idx, window in enumerate(windows):
            frames = window["frames"]
            start_time = window["start"]
            end_time = window["end"]

            if not frames:
                print(f"\n🔕 在线窗口 {idx+1}/{len(windows)} | {start_time:.2f}-{end_time:.2f}秒 | 无有效帧，跳过")
                continue

            print(f"\n🪟 在线窗口 {idx+1}/{len(windows)} | 时间范围: {start_time:.2f}-{end_time:.2f}秒")
            print(f"   采样帧数: {len(frames)} | 固定判断动作数: {len(actions_to_detect)} (前{self.max_action_num}个)")

            try:
                prompt = self.build_multi_action_prompt(window, actions_to_detect)
                messages = [
                    {
                        "role": "user",
                        "content": ([{'type': 'image', 'image': img} for img in frames] +
                                   [{'type': 'text', 'text': prompt}])
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        num_beams=1,
                    )

                output_text = self.processor.batch_decode(
                    [generated_ids[0][len(inputs.input_ids[0]):]],
                    skip_special_tokens=True
                )[0]

                start_idx = output_text.find("{")
                end_idx = output_text.rfind("}")
                if start_idx == -1 or end_idx == -1:
                    print(f"   ❌ JSON解析失败 | 模型输出: {output_text[:150]}...")
                    continue

                result = json.loads(output_text[start_idx:end_idx+1])
                action_results = result.get("actions", {})

                window_raw = {
                    "window_idx": idx,
                    "start": start_time,
                    "end": end_time,
                    "actions": action_results
                }
                window_raw_results.append(window_raw)

                print(f"   ✅ 多动作判断完成 | 解析到 {len(action_results)} 个动作结果 (预期{len(actions_to_detect)}个)")

                for action in actions_to_detect:
                    action_class = action['class']
                    if action_class not in action_results:
                        action_results[action_class] = {
                            "happens": False,
                            "confidence": 0.0,
                            "explanation": "Not detected"
                        }

                for action in actions_to_detect:
                    action_class = action['class']
                    action_result = action_results.get(action_class, {
                        "happens": False,
                        "confidence": 0.0,
                        "explanation": "Not detected"
                    })

                    happens = action_result.get("happens", False)
                    confidence = action_result.get("confidence", 0.0)
                    explanation = action_result.get("explanation", "")[:20]

                    act_desc = self.action_classes.get(action_class, f"Unknown {action_class}")
                    status = "✅ 检测到" if happens else "❌ 未检测到"
                    print(f"   - {action_class}: {status} | 置信度: {confidence:.3f} | {act_desc[:30]}")

                    if happens and confidence >= self.conf_threshold:
                        multi_action_results[action_class].append({
                            "start": start_time,
                            "end": end_time,
                            "confidence": confidence,
                            "explanation": explanation,
                            "window_idx": idx,
                            "window_indices": [idx]
                        })

            except Exception as e:
                print(f"   ❌ 窗口处理失败: {str(e)[:100]}")
                continue

        print(f"\n====================================================")
        print(f"🔗 开始合并同标签的连续有效时间段 (阈值: {self.merge_time_threshold}秒)")
        print(f"====================================================")

        merged_multi_action_results = {}
        for action_class, segments in multi_action_results.items():
            merged_segments = merge_contiguous_segments(segments, self.merge_time_threshold)
            merged_multi_action_results[action_class] = merged_segments

            act_desc = self.action_classes.get(action_class, f"Unknown {action_class}")
            print(f"\n🔍 {action_class}: {act_desc}")
            print(f"   合并前时间段数: {len(segments)} | 合并后: {len(merged_segments)}")

        gt_segments = {a["class"]: {"start": a["start"], "end": a["end"]} for a in actions_to_detect}
        model_segments = {}
        for ac, segs in merged_multi_action_results.items():
            model_segments[ac] = [{"start": s["start"], "end": s["end"], "confidence": s["confidence"]} for s in segs]

        evaluation_metrics = evaluate_video_complete(
            video_id=video_id,
            window_results=window_raw_results,
            pred_segments=model_segments,
            gt_segments=gt_segments,
            tiou_thresholds=self.tiou_thresholds
        )

        final_result = {
            "video_id": video_id,
            "ground_truth": gt_segments,
            "model_detections": model_segments,
            "window_raw_results": window_raw_results,
            "processing_info": {
                "window_size": self.window_size,
                "stride": self.stride,
                "conf_threshold": self.conf_threshold,
                "max_action_num": self.max_action_num,
                "actual_action_num": len(actions_to_detect),
                "tiou_thresholds": self.tiou_thresholds
            },
            "evaluation": evaluation_metrics
        }

        print(f"\n====================================================")
        print(f"📊 视频 {video_id} 处理完成")
        print(f"====================================================")
        return final_result

    def process_batch(self, split='test', max_videos=None, output_file='charades_grounding_batch_baseline.json'):
        # 获取所有视频ID
        all_video_ids = list(self.dataset_handler.train_annotations.keys()) if split == 'train' else list(self.dataset_handler.test_annotations.keys())
        if max_videos:
            all_video_ids = all_video_ids[:max_videos]

        # ========== 过滤掉没有动作标签的视频 ==========
        filtered_video_ids = []
        skipped_empty_videos = []
        
        for video_id in all_video_ids:
            annotation = self.dataset_handler.get_video_annotation(video_id)
            if annotation and annotation.get('actions') and len(annotation['actions']) > 0:
                filtered_video_ids.append(video_id)
            else:
                skipped_empty_videos.append(video_id)
        
        print(f"\n====================================================")
        print(f"🚀 开始批量处理视频 (BASELINE 无记忆版本 | 对齐记忆版评估格式)")
        print(f"📁 数据集划分: {split} | 原始数量: {len(all_video_ids)}")
        print(f"✅ 有效视频(有动作标签): {len(filtered_video_ids)}")
        print(f"⏭️  跳过视频(无动作标签): {len(skipped_empty_videos)}")
        
        if skipped_empty_videos and len(skipped_empty_videos) <= 10:
            print(f"   跳过的视频ID: {skipped_empty_videos}")
        elif skipped_empty_videos:
            print(f"   跳过的视频ID(前10个): {skipped_empty_videos[:10]}...")
        
        print(f"🎯 tIoU评估阈值: {self.tiou_thresholds}")
        print(f"====================================================")
        # =====================================================

        batch_results = []
        for i, video_id in enumerate(filtered_video_ids):
            print(f"\n🔢 批量进度: {i+1}/{len(filtered_video_ids)}")
            res = self.process_single_video(video_id)
            if res:
                batch_results.append(res)

        print(f"\n====================================================")
        print(f"📊 开始批量评估 (共{len(batch_results)}个视频)")
        print(f"====================================================")

        batch_eval = evaluate_batch_complete(batch_results, self.tiou_thresholds)

        print(f"\n🎯 BASELINE 批量评估结果 (对齐记忆版输出):")
        print(f"   - 视频总数: {batch_eval['total_videos']}")
        print(f"   - 平均窗口级F1: {batch_eval['summary']['mean_window_f1']:.4f}")
        print(f"   - 平均事件级F1: {batch_eval['summary']['mean_event_f1']:.4f}")
        print(f"   - 平均tIoU: {batch_eval['summary']['mean_event_tiou']:.4f}")
        for t in self.tiou_thresholds:
            k = f"tiou_{t}_mean_event_f1"
            if k in batch_eval['summary']:
                print(f"   - tIoU@{t} F1: {batch_eval['summary'][k]:.4f}")

        print(f"\n📋 整体统计:")
        wl = batch_eval['detailed_summary']['window_level']
        el = batch_eval['detailed_summary']['event_level']
        print(f"   窗口级 | P: {wl['overall_precision']:.4f} R: {wl['overall_recall']:.4f} F1: {wl['overall_f1']:.4f}")
        print(f"   事件级 | P: {el['overall_precision']:.4f} R: {el['overall_recall']:.4f} F1: {el['overall_f1']:.4f}")
        for t in self.tiou_thresholds:
            d = batch_eval['detailed_summary'][f'tiou_{t}']
            print(f"   tIoU@{t} | P: {d['overall_precision']:.4f} R: {d['overall_recall']:.4f} F1: {d['overall_f1']:.4f}")

        final_out = {
            "batch_info": {
                "method": "baseline_without_memory",
                "window_size": self.window_size,
                "stride": self.stride,
                "max_action_num": self.max_action_num,
                "tiou_thresholds": self.tiou_thresholds
            },
            "video_results": batch_results,
            "batch_evaluation": batch_eval
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_out, f, indent=4, ensure_ascii=False)

        print(f"\n🎉 批量完成 | 输出: {output_file}")
        return final_out

# ======================== 主函数 ========================
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Charades BASELINE 无记忆版本（对齐记忆版评估）')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--video_dir', type=str, required=True, help='视频目录')
    parser.add_argument('--annotation_dir', type=str, required=True, help='标注目录')
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_videos', type=int, default=2)
    parser.add_argument('--output_file', type=str, default='charades_baseline_aligned.json')
    parser.add_argument('--max_action_num', type=int, default=5)
    parser.add_argument('--tiou_thresholds', type=str, default='0.3,0.5')
    args = parser.parse_args()

    tiou_thresholds = [float(x.strip()) for x in args.tiou_thresholds.split(',')]
    grounding = CharadesVideoGrounding(
        model_path=args.model_path,
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        device_map=args.device_map,
        max_action_num=args.max_action_num,
        tiou_thresholds=tiou_thresholds
    )

    grounding.process_batch(
        split=args.split,
        max_videos=args.max_videos,
        output_file=args.output_file
    )

    elapsed_str, _ = get_elapsed_time(start_time)
    print(f"\n⏱️ 总运行时间: {elapsed_str}")

if __name__ == "__main__":
    main()