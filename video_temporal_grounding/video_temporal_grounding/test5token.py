import torch
import json
import warnings
import cv2
import numpy as np
import re
import argparse
import time
from pathlib import Path
from collections import deque
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# 导入已有的数据集处理器
from dataset import CharadesDatasetHandler
from evaluation_metrics import (
    calculate_tiou,
    calculate_precision_recall_f1,
    evaluate_window_level,
    evaluate_event_level,
    evaluate_video_complete,
    evaluate_batch_complete,
    merge_contiguous_segments as eval_merge_segments
)

# 忽略不必要的警告
warnings.filterwarnings('ignore')

def get_elapsed_time(start_time):
    """计算运行时间"""
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

# ======================== 【消融修改】删除前4个（视觉2+时序2），仅保留后3个Token：state_action_scene ========================
def extract_key_tokens_from_explanation(explanation, action_class, action_desc, state="none"):
    """
    7个Token固定原始格式顺序：
    [1视觉1]_[2视觉2]_[3时序1]_[4时序2]_[5state]_[6action]_[7scene]
    
    本次消融严格要求：
    1. 前4位（视觉1、视觉2、时序1、时序2）**全部删除，强制固定为none**
    2. 保留后3位完整有效信息：第5位state + 第6位action + 第7位scene
    3. 严格保持7位下划线格式，完全兼容原有所有解析、记忆、打印代码
    4. 其余所有代码结构、衰减、评估全部一丝不动
    """
    # ========== 前4个：视觉、时序全部删除，统一填none ==========
    vis1 = "none"
    vis2 = "none"
    temp1 = "none"
    temp2 = "none"

    # ========== 后3个：完整保留（本次消融保留部分）==========
    # 第5位：状态state 直接使用传入参数
    token_state = state

    # 第6位：动作action 直接使用动作类别
    token_action = action_class

    # 第7位：场景scene 从原始动作描述/标注信息提取
    clean_text = re.sub(r'[^\w\s]', '', explanation.lower()).strip()
    scene_keywords = ['room', 'house', 'kitchen', 'bedroom', 'living', 'office', 'outdoor', 'indoor',
                      'floor', 'wall', 'table', 'desk', 'car', 'street', 'building']
    token_scene = "none"
    for word in clean_text.split():
        if word in scene_keywords:
            token_scene = word
            break

    # 严格7位拼接格式，和原版结构完全对齐
    return f"{vis1}_{vis2}_{temp1}_{temp2}_{token_state}_{token_action}_{token_scene}"

# ======================== 动态历史记忆库模块（仅保留state衰减，其余结构100%原样保留） ========================
class DynamicActionMemory:
    def __init__(self, memory_size=5, time_horizon=10.0):
        self.action_memories = {}
        self.memory_size = memory_size
        self.time_horizon = time_horizon
        # 【完全保留你原始全部非对称衰减配置，一丝不动】
        self.decay_config = {
            "start": 0.85,     # 起始状态衰减慢
            "continue": 0.80,  # 持续状态衰减慢
            "end": 0.35,       # 结束状态衰减快
            "none": 0.10,      # 降低none衰减率，解决负面记忆泛滥压制检测
            "default": 0.85    # 默认衰减率
        }
        self.min_conf_threshold = 0.10  # 最小置信度阈值

    def update_memory(self, action_class, window_data):
        """
        更新记忆库：存储时间、置信度、state、7位token格式key_tokens
        【代码结构、字段完全和原版保持一致】
        """
        if action_class not in self.action_memories:
            self.action_memories[action_class] = deque(maxlen=self.memory_size)
        
        # 提取7位token
        key_tokens = window_data.get("key_tokens", "none_none_none_none_none_none_none")
        state = window_data.get("state", "none")
        
        # 构造summary格式完全不变
        summary = f"{state}_{key_tokens}_conf{window_data['confidence']:.2f}"
        
        self.action_memories[action_class].append({
            "start": window_data['start'],
            "end": window_data['end'],
            "conf": window_data['confidence'],
            "state": state,
            "key_tokens": key_tokens,
            "summary": summary
        })

    def update_negative_memory(self, action_class, current_time, action_desc):
        """
        更新负面记忆：仅state=none，严格消融，无任何额外信息
        """
        if action_class not in self.action_memories:
            self.action_memories[action_class] = deque(maxlen=self.memory_size)
        
        # 强制生成7位格式token
        key_tokens = extract_key_tokens_from_explanation(
            explanation="",
            action_class=action_class,
            action_desc=action_desc,
            state="none"
        )
        negative_memory = {
            "start": current_time,
            "end": current_time + 1.0,
            "conf": 0.05,
            "state": "none",
            "key_tokens": key_tokens,
            "summary": f"none_{key_tokens}_conf0.05"
        }
        self.action_memories[action_class].append(negative_memory)

    def get_memory_prompt(self, action_class, current_time):
        """
        生成记忆Prompt：完全适配原有打印格式，代码完全不动
        """
        if action_class not in self.action_memories or not self.action_memories[action_class]:
            return "No prior history in earlier segments."
        
        history_segments = []
        for m in list(self.action_memories[action_class]):
            time_diff = current_time - m['end']
            if time_diff > self.time_horizon:
                continue
            
            # 完全保留原始非对称衰减计算逻辑
            decay_rate = self.decay_config.get(m['state'], self.decay_config["default"])
            decayed_conf = m['conf'] * (decay_rate ** max(0, time_diff))
            
            if decayed_conf >= self.min_conf_threshold:
                # 兼容原有7token解析打印格式
                tokens = m['key_tokens'].split('_')
                vis1 = tokens[0] if len(tokens)>=1 else "none"
                vis2 = tokens[1] if len(tokens)>=2 else "none"
                temp1 = tokens[2] if len(tokens)>=3 else "none"
                temp2 = tokens[3] if len(tokens)>=4 else "none"
                state_token = tokens[4] if len(tokens)>=5 else "none"
                act_token = tokens[5] if len(tokens)>=6 else "none"
                scene_token = tokens[6] if len(tokens)>=7 else "none"
                
                history_segments.append(
                    f"[{m['start']:.1f}-{m['end']:.1f}s, {m['state']}, "
                    f"Vis:{vis1}/{vis2}, "
                    f"Temp:{temp1}/{temp2}, "
                    f"Act:{act_token}, Eff.Conf:{decayed_conf:.2f}]"
                )
        
        return " -> ".join(history_segments) if history_segments else "Recent history has faded."

# ======================== 视频采样模块（【完全原样不动】） ========================
def sample_video_frames_sliding_window(video_path, window_size=1, stride=1, num_frames=12):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    windows = []
    start_time = 0.0
    
    while start_time < duration:
        end_time = min(start_time + window_size, duration)
        current_window_frames = []
        
        # 前4帧：历史记忆帧
        history_start = max(0, start_time - 2)
        history_time_points = np.linspace(history_start, max(history_start, start_time - 0.1), 4)
        
        # 后8帧：当前窗口帧
        current_time_points = np.linspace(start_time, end_time, 8)
        
        all_time_points = list(history_time_points) + list(current_time_points)
        all_frame_indices = [min(int(t * fps), total_frames - 1) for t in all_time_points]
        
        for idx in all_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                current_window_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                if current_window_frames:
                    current_window_frames.append(current_window_frames[-1])
                else:
                    current_window_frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
        
        windows.append({
            "start": start_time, 
            "end": end_time, 
            "frames": current_window_frames,
            "timestamps": all_time_points,
            "frame_indices": all_frame_indices,
            "frame_types": ["history"]*4 + ["current"]*8
        })
        
        start_time += stride
    
    cap.release()
    print(f"✅ 视频采样完成 - 总时长: {duration:.2f}秒 | 窗口大小: {window_size}秒 | 步长: {stride}秒 | 采样帧数: 12 (4历史+8当前) | 总窗口数: {len(windows)}")
    return windows, duration

# ======================== 工具函数：时间段合并（【完全原样不动】） ========================
def merge_contiguous_segments(segments, time_threshold=0.1):
    """合并连续的时间段"""
    return eval_merge_segments(segments, time_threshold)

# ======================== 增强型视频定位主模块（【主体代码全部原样不动】） ========================
class CharadesVideoGrounding:
    def __init__(self, model_path, video_dir, annotation_dir, device_map="auto",
                 window_size=1, stride=1, num_frames=12, conf_threshold=0.6, max_action_num=6,
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
        self.memory_manager = DynamicActionMemory(memory_size=5, time_horizon=10.0)
        
        self.window_size = window_size
        self.stride = stride
        self.num_frames = num_frames
        self.conf_threshold = conf_threshold
        self.merge_time_threshold = 0.1
        self.max_action_num = max_action_num
        self.tiou_thresholds = tiou_thresholds
        self.action_classes = self._load_action_classes(annotation_dir)
        
        print(f"🎯 消融实验配置: 删除【视觉2+时序2】，仅保留后3Token (state_action_scene)")

    def _load_action_classes(self, annotation_dir):
        """加载动作类别文件"""
        class_file = Path(annotation_dir) / "Charades_v1_classes.txt"
        classes = {}
        if class_file.exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) >= 2: 
                        classes[parts[0]] = parts[1]
        else:
            print(f"⚠️  动作类别文件未找到: {class_file}")
        return classes

    def get_video_full_info(self, video_id):
        """获取视频完整信息"""
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
        """构建多动作检测的Prompt【原文完全原样不动，无任何修改】"""
        start = window_info["start"]
        end = window_info["end"]
        
        frame_desc = []
        for i in range(4):
            frame_time = window_info['timestamps'][i]
            frame_desc.append(f"Frame {i+1}: Historical memory frame (from {frame_time:.2f}s, within past 2 seconds)")
        
        for i in range(4, 12):
            frame_num = i + 1
            frame_time = window_info['timestamps'][i]
            if i == 4:
                frame_desc.append(f"Frame {frame_num}: Current segment START ({frame_time:.2f}s)")
            elif i == 11:
                frame_desc.append(f"Frame {frame_num}: Current segment END ({frame_time:.2f}s)")
            else:
                frame_desc.append(f"Frame {frame_num}: Current segment ({frame_time:.2f}s)")
        
        time_desc = "\n".join(frame_desc)
        
        memory_context_list = []
        for action in actions_to_detect:
            a_class = action.get('class', 'unknown')
            a_desc = self.action_classes.get(a_class, 'Unknown action')
            history = self.memory_manager.get_memory_prompt(a_class, start)
            memory_context_list.append(
                f"- {a_class}: {a_desc} | History: {history} (annotated time: {action['start']:.2f}-{action['end']:.2f}s)"
            )
        
        action_list_str = "\n".join(memory_context_list)
        
        prompt = f"""
You are a professional video understanding assistant specialized in real-time temporal action localization.
You need to analyze THIS 1-second video segment and judge MULTIPLE actions simultaneously.

=== VIDEO FRAME INFORMATION (CRITICAL ORDER) ===
{time_desc}

IMPORTANT FRAME ORDER: Frames 1-4 are HISTORICAL MEMORY FRAMES sampled from the past 2 seconds (in chronological order).
Frames 5-12 are HIGH-SAMPLING FRAMES of the current 1-second segment ({start:.2f}-{end:.2f}s, in chronological order).

You MUST analyze frames in this specific order to understand temporal progression:
1. First examine HISTORICAL FRAMES (1-4) to understand what was happening 0-2 seconds ago
2. Then examine CURRENT FRAMES (5-12) to detect what is happening now

=== ACTIONS TO DETECT (with semantic history) ===
{action_list_str}

=== TASK ===
For EACH action in the list above, determine:
1. Whether it happens in THIS 1-second segment (true/false)
2. Confidence score (0-1, 1=absolutely certain)
3. State: 
   - 'start': Action is NOT present in historical frames (1-4) but appears in current frames (5-12)
   - 'continue': Action is present in both historical frames (1-4) and current frames (5-12)
   - 'end': Action is present in historical frames (1-4) but disappears in current frames (5-12)
   - 'none': Action is not present in either historical or current frames
4. Brief explanation (max 20 words) - Include visual evidence from historical vs current comparison

=== OUTPUT FORMAT ===
Return ONLY a compact valid JSON object with NO extra text, NO line breaks:
{{"actions":{{"ACTION_CLASS":{{"happens":true/false,"state":"start/continue/end/none","confidence":0.0-1.0,"explanation":"brief reason"}}}},"window_info":{{"start":{start:.2f},"end":{end:.2f}}}}}
"""
        return prompt

    def process_single_video(self, video_id):
        """处理单个视频【主体全代码原样保留，仅新增每窗口原始结果打印，方便排查】"""
        video_info = self.get_video_full_info(video_id)
        if not video_info:
            print(f"❌ 无法找到视频 {video_id} 的完整信息")
            return None
        
        # ========== 【保留】检测视频无动作标签，直接跳过 ==========
        original_actions = video_info.get('actions', [])
        if not original_actions or len(original_actions) == 0:
            print(f"⏭️  跳过视频 {video_id}: 无动作标签")
            return None
        
        video_path = video_info['path']
        duration = video_info['duration']
        original_action_count = len(original_actions)
        
        if original_action_count >= self.max_action_num:
            actions_to_detect = original_actions[:self.max_action_num]
            select_note = f"取前{self.max_action_num}个"
        else:
            actions_to_detect = original_actions.copy()
            select_note = f"全部{original_action_count}个（不足{self.max_action_num}个）"

        print(f"\n====================================================")
        print(f"📹 开始处理视频: {video_id}")
        print(f"📍 场景: {video_info['scene']} | 时长: {duration:.2f}秒 | 划分: {video_info['split']}")
        print(f"⚙️  滑窗配置: 窗口={self.window_size}秒 | 步长={self.stride}秒 | 采样帧数=12 (4历史记忆帧 + 8当前帧)")
        print(f"📝 待检测动作数: {len(actions_to_detect)} (原始{original_action_count}个，{select_note})")
        print(f"🎯 tIoU评估阈值: {self.tiou_thresholds}")
        print(f"🔄 非对称衰减配置: start=0.85 | continue=0.80 | end=0.35 | none=0.10")
        print(f"🎨 消融配置：删除视觉+时序Token，仅保留【state_action_scene】后3个Token")
        
        for idx, act in enumerate(actions_to_detect):
            act_class = act.get('class', 'unknown')
            act_desc = self.action_classes.get(act_class, '未知动作')
            act_start = act.get('start', 0.0)
            act_end = act.get('end', 0.0)
            print(f"   - 动作{idx+1}: {act_class} | {act_desc} | 标注时间: {act_start:.2f}-{act_end:.2f}秒")
        print(f"====================================================")
        
        windows, _ = sample_video_frames_sliding_window(
            video_path,
            window_size=self.window_size,
            stride=self.stride,
            num_frames=self.num_frames
        )
        
        multi_action_results = {a.get('class', f'empty_{id(a)}'): [] for a in actions_to_detect}
        window_raw_results = []
        
        self.memory_manager.action_memories = {}
        
        for idx, window in enumerate(windows):
            frames = window["frames"]
            start_time = window["start"]
            end_time = window["end"]
            window_action_preds = {}
            
            if not frames:
                print(f"\n🔕 在线窗口 {idx+1}/{len(windows)} | {start_time:.2f}-{end_time:.2f}秒 | 无有效帧，跳过")
                for action in actions_to_detect:
                    action_class = action.get('class', f'empty_{id(action)}')
                    action_desc = self.action_classes.get(action_class, 'Unknown')
                    self.memory_manager.update_negative_memory(action_class, start_time, action_desc)
                    window_action_preds[action_class] = {
                        "happens": False,
                        "state": "none",
                        "confidence": 0.0,
                        "explanation": "No valid frames"
                    }
                window_raw_results.append({"window_idx": idx, "start": start_time, "end": end_time, "actions": window_action_preds})
                continue
            
            print(f"\n🪟 在线窗口 {idx+1}/{len(windows)} | {start_time:.2f}-{end_time:.2f}秒")
            print(f"   采样帧数: {len(frames)} (4历史记忆帧+8当前帧) | 并行判断动作数: {len(actions_to_detect)}")
            
            try:
                prompt = self.build_multi_action_prompt(window, actions_to_detect)
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": img} for img in frames] + [{"type": "text", "text": prompt}]
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
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=0.0,
                        num_beams=1
                    )
                
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
                
                start_idx = output_text.find("{")
                end_idx = output_text.rfind("}")
                if start_idx == -1 or end_idx == -1:
                    print(f"   ❌ JSON解析失败 | 未找到完整JSON | 输出片段: {output_text[:200]}...")
                    for action in actions_to_detect:
                        action_class = action.get('class', f'empty_{id(action)}')
                        action_desc = self.action_classes.get(action_class, 'Unknown')
                        self.memory_manager.update_negative_memory(action_class, start_time, action_desc)
                        window_action_preds[action_class] = {
                            "happens": False, "state": "none", "confidence": 0.0, "explanation": "JSON parse failed"
                        }
                    window_raw_results.append({"window_idx": idx, "start": start_time, "end": end_time, "actions": window_action_preds})
                    continue
                
                json_str = output_text[start_idx:end_idx+1].replace("\n", "").replace("\t", "").replace("  ", " ").strip()
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON解析失败: {e}")
                    for action in actions_to_detect:
                        action_class = action.get('class', f'empty_{id(action)}')
                        self.memory_manager.update_negative_memory(action_class, start_time, self.action_classes.get(action_class, ''))
                        window_action_preds[action_class] = {"happens": False, "state": "none", "confidence": 0.0, "explanation": "JSON error"}
                    window_raw_results.append({"window_idx": idx, "start": start_time, "end": end_time, "actions": window_action_preds})
                    continue
                
                action_results = result.get("actions", {})
                print(f"   ✅ 多动作判断完成 | 解析到 {len(action_results)} 个动作结果 | 需补齐到 {len(actions_to_detect)} 个")
                
                for action in actions_to_detect:
                    action_class = action.get('class', f'empty_{id(action)}')
                    if action_class not in action_results:
                        action_results[action_class] = {
                            "happens": False,
                            "state": "none",
                            "confidence": 0.0,
                            "explanation": "Not detected in this segment"
                        }
                
                for action in actions_to_detect:
                    action_class = action.get('class', f'empty_{id(action)}')
                    action_result = action_results.get(action_class, {})
                    
                    happens = action_result.get("happens", False)
                    confidence = action_result.get("confidence", 0.0)
                    state = action_result.get("state", "none")
                    explanation = action_result.get("explanation", "")
                    
                    window_action_preds[action_class] = {
                        "happens": happens,
                        "state": state,
                        "confidence": confidence,
                        "explanation": explanation
                    }
                    
                    action_desc = self.action_classes.get(action_class, "Unknown")
                    # 调用新版消融token函数
                    key_tokens = extract_key_tokens_from_explanation(
                        explanation=explanation, 
                        action_class=action_class, 
                        action_desc=action_desc,
                        state=state
                    )
                    
                    # 每动作原始结果打印
                    print(f"   - {action_class}: 原始输出 happens={happens}, state={state}, 原始置信度={confidence:.3f}")
                    
                    status = "✅ 检测到" if happens else "❌ 未检测到"
                    state_tag = f"[STATE:{state.upper()}]" if state in ["start", "continue"] else f"[{state}]"
                    decay_tag = f"[SLOW DECAY]" if state in ["start", "continue"] else f"[FAST DECAY]"
                    
                    if happens and confidence >= self.conf_threshold:
                        print(f"   ✅ 满足阈值({self.conf_threshold})，纳入有效预测 & 更新记忆")
                        self.memory_manager.update_memory(action_class, {
                            "start": start_time,
                            "end": end_time,
                            "confidence": confidence,
                            "state": state,
                            "key_tokens": key_tokens,
                            "explanation": explanation
                        })
                        multi_action_results[action_class].append({
                            "start": start_time,
                            "end": end_time,
                            "confidence": confidence,
                            "state": state,
                            "key_tokens": key_tokens,
                            "window_idx": idx,
                            "window_indices": [idx]
                        })
                    else:
                        print(f"   ❌ 不满足阈值，更新负面记忆")
                        self.memory_manager.update_negative_memory(action_class, start_time, action_desc)
                
                window_raw_results.append({"window_idx": idx, "start": start_time, "end": end_time, "actions": window_action_preds})
            
            except Exception as e:
                print(f"   ❌ 窗口处理失败: {str(e)[:100]}")
                for action in actions_to_detect:
                    action_class = action.get('class', f'empty_{id(action)}')
                    self.memory_manager.update_negative_memory(action_class, start_time, self.action_classes.get(action_class, ''))
                    window_action_preds[action_class] = {"happens": False, "state": "none", "confidence": 0.0, "explanation": f"Error: {str(e)[:50]}"}
                window_raw_results.append({"window_idx": idx, "start": start_time, "end": end_time, "actions": window_action_preds})
                continue
        
        print(f"\n====================================================")
        print(f"🔗 开始合并同标签的连续有效时间段 (阈值: {self.merge_time_threshold}秒)")
        print(f"====================================================")
        
        merged_multi_action_results = {}
        for action_class, segments in multi_action_results.items():
            merged_segments = merge_contiguous_segments(segments, self.merge_time_threshold)
            merged_multi_action_results[action_class] = merged_segments
            
            action_desc = self.action_classes.get(action_class, f"Unknown {action_class}")
            print(f"\n🔍 {action_class}: {action_desc}")
            print(f"   合并前时间段数: {len(segments)} | 合并后: {len(merged_segments)}")
            
            if merged_segments:
                for i, seg in enumerate(merged_segments):
                    print(f"   {i+1}. {seg['start']:.2f}-{seg['end']:.2f}秒 (置信度: {seg['confidence']:.3f})")
            else:
                print(f"   ❌ 无有效时间段")
        
        gt_segments = {}
        for action in actions_to_detect:
            action_class = action.get('class', f'empty_{id(action)}')
            action_desc = self.action_classes.get(action_class, "Unknown")
            gt_segments[action_class] = {
                "start": round(action.get('start', 0.0), 2),
                "end": round(action.get('end', 0.0), 2),
                "description": action_desc
            }
        
        model_segments = {}
        for action_class, segments in merged_multi_action_results.items():
            model_segments[action_class] = []
            for seg in segments:
                model_segments[action_class].append({
                    "start": round(seg['start'], 2),
                    "end": round(seg['end'], 2),
                    "confidence": round(seg['confidence'], 3),
                    "window_indices": seg.get("window_indices", []),
                    "window_idx": seg.get("window_idx", -1),
                    "key_tokens_7": seg.get("key_tokens", "")
                })
        
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
                "original_action_num": original_action_count,
                "tiou_thresholds": self.tiou_thresholds,
                "token_config": "ABLATION: 删除视觉2+时序2，仅保留后3Token (state_action_scene)"
            },
            "evaluation": evaluation_metrics
        }
        
        return final_result

    def process_batch(self, split='test', max_videos=None, output_file='charades_grounding_batch_state_act_scene.json'):
        """批量处理视频【全部原样保留，仅修改配置打印信息】"""
        all_video_ids = list(self.dataset_handler.train_annotations.keys()) if split == 'train' else list(self.dataset_handler.test_annotations.keys())
        if max_videos:
            all_video_ids = all_video_ids[:max_videos]
        
        # ========== 批量前置过滤：无动作标签视频全部跳过 ==========
        filtered_video_ids = []
        skipped_empty_videos = []
        
        for video_id in all_video_ids:
            annotation = self.dataset_handler.get_video_annotation(video_id)
            if annotation and annotation.get('actions') and len(annotation['actions']) > 0:
                filtered_video_ids.append(video_id)
            else:
                skipped_empty_videos.append(video_id)
        
        print(f"\n====================================================")
        print(f"🚀 开始批量处理视频")
        print(f"📁 数据集划分: {split} | 原始数量: {len(all_video_ids)}")
        print(f"✅ 有效视频(有动作标签): {len(filtered_video_ids)}")
        print(f"⏭️  跳过视频(无动作标签): {len(skipped_empty_videos)}")
        
        if skipped_empty_videos and len(skipped_empty_videos) <= 10:
            print(f"   跳过的视频ID: {skipped_empty_videos}")
        elif skipped_empty_videos:
            print(f"   跳过的视频ID(前10个): {skipped_empty_videos[:10]}...")
        
        print(f"🎯 tIoU评估阈值: {self.tiou_thresholds}")
        print(f"🔄 非对称衰减配置: start=0.85 | continue=0.80 | end=0.35 | none=0.10")
        print(f"🎨 消融配置：删除视觉+时序Token，仅保留【state_action_scene】后3个Token")
        print(f"====================================================")
        
        batch_results = []
        for i, video_id in enumerate(filtered_video_ids):
            print(f"\n🔢 批量进度: {i+1}/{len(filtered_video_ids)}")
            result = self.process_single_video(video_id)
            if result:
                batch_results.append(result)
        
        print(f"\n====================================================")
        print(f"📊 开始批量评估 (共{len(batch_results)}个视频)")
        print(f"🎯 tIoU评估阈值: {self.tiou_thresholds}")
        print(f"====================================================")
        
        batch_evaluation = evaluate_batch_complete(batch_results=batch_results, tiou_thresholds=self.tiou_thresholds)
        
        print(f"\n🎯 批量评估结果汇总:")
        print(f"   - 处理视频总数: {batch_evaluation['total_videos']}")
        print(f"   - 平均窗口级F1分数: {batch_evaluation['summary']['mean_window_f1']:.4f}")
        print(f"   - 平均事件级F1分数: {batch_evaluation['summary']['mean_event_f1']:.4f}")
        print(f"   - 平均事件级tIoU: {batch_evaluation['summary']['mean_event_tiou']:.4f}")
        for tiou in self.tiou_thresholds:
            key = f'tiou_{tiou}_mean_event_f1'
            if key in batch_evaluation['summary']:
                print(f"   - tIoU@{tiou} 事件级F1: {batch_evaluation['summary'][key]:.4f}")
        
        print(f"\n📋 详细统计:")
        print(f"   窗口级 - 精确率: {batch_evaluation['detailed_summary']['window_level']['overall_precision']:.4f} | 召回率: {batch_evaluation['detailed_summary']['window_level']['overall_recall']:.4f} | F1: {batch_evaluation['detailed_summary']['window_level']['overall_f1']:.4f}")
        print(f"   事件级 - 精确率: {batch_evaluation['detailed_summary']['event_level']['overall_precision']:.4f} | 召回率: {batch_evaluation['detailed_summary']['event_level']['overall_recall']:.4f} | F1: {batch_evaluation['detailed_summary']['event_level']['overall_f1']:.4f}")
        
        final_batch_result = {
            "batch_info": {
                "split": split,
                "max_videos": max_videos,
                "original_video_count": len(all_video_ids),
                "filtered_video_count": len(filtered_video_ids),
                "skipped_empty_videos": len(skipped_empty_videos),
                "window_size": self.window_size,
                "stride": self.stride,
                "conf_threshold": self.conf_threshold,
                "max_action_num": self.max_action_num,
                "tiou_thresholds": self.tiou_thresholds,
                "token_config": "ABLATION: 删除视觉2+时序2，仅保留后3Token (state_action_scene)"
            },
            "video_results": batch_results,
            "batch_evaluation": batch_evaluation
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_batch_result, f, indent=4, ensure_ascii=False)
        
        print(f"\n====================================================")
        print(f"🎉 批量处理+评估完成！")
        print(f"💾 结果已保存到: {output_file}")
        print(f"====================================================")
        
        return final_batch_result

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Charades视频动作定位 (消融-仅保留state_action_scene后3Token)')
    parser.add_argument('--model_path', type=str, required=True, help='Qwen3VL模型路径')
    parser.add_argument('--video_dir', type=str, required=True, help='Charades视频文件夹路径')
    parser.add_argument('--annotation_dir', type=str, required=True, help='Charades标注文件夹路径')
    parser.add_argument('--device_map', type=str, default='auto', help='设备映射（auto/cpu/cuda）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='数据集划分')
    parser.add_argument('--max_videos', type=int, default=2, help='批量处理的最大视频数')
    parser.add_argument('--output_file', type=str, default='charades_grounding_batch_state_act_scene.json', help='批量结果输出文件')
    parser.add_argument('--window_size', type=int, default=1, help='滑动窗口大小（秒）')
    parser.add_argument('--stride', type=int, default=1, help='滑动窗口步长（秒）')
    parser.add_argument('--num_frames', type=int, default=12, help='采样帧数（固定12帧）')
    parser.add_argument('--conf_threshold', type=float, default=0.6, help='置信度阈值')
    parser.add_argument('--max_action_num', type=int, default=6, help='每视频最多检测动作数（默认6）')
    parser.add_argument('--tiou_thresholds', type=str, default='0.3,0.5', help='tIoU评估阈值，多个值用逗号分隔')
    
    args = parser.parse_args()
    
    tiou_thresholds = [float(x.strip()) for x in args.tiou_thresholds.split(',')]
    
    grounding = CharadesVideoGrounding(
        model_path=args.model_path,
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        device_map=args.device_map,
        window_size=args.window_size,
        stride=args.stride,
        num_frames=args.num_frames,
        conf_threshold=args.conf_threshold,
        max_action_num=args.max_action_num,
        tiou_thresholds=tiou_thresholds
    )
    
    results = grounding.process_batch(
        split=args.split,
        max_videos=args.max_videos,
        output_file=args.output_file
    )
    
    elapsed_str, _ = get_elapsed_time(start_time)
    print(f"\n⏱️  程序总运行时长: {elapsed_str}")

if __name__ == "__main__":
    main()