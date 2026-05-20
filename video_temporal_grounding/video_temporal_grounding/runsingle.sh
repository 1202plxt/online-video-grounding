#!/bin/bash
# Charades视频动作定位运行脚本（BASELINE对齐版｜单视频测试）
# 使用说明：修改下方的路径参数后，执行 chmod +x run1.sh && ./run1.sh

# ======================== 配置参数（根据你的环境修改） ========================
# 模型路径
MODEL_PATH="/root/autodl-tmp/Qwen3-VL-8B-Instruct"
# 视频文件夹路径
VIDEO_DIR="/root/autodl-tmp/Charades_v1_480"
# 标注文件夹路径
ANNOTATION_DIR="/root/autodl-tmp/Charades"
# 数据集划分（train/test）
SPLIT="test"

# --- 新增：指定要测试的视频索引 ---
# 19 代表测试过滤后的第 20 个视频
TARGET_INDEX=19

# 批量处理的最大视频数（当 TARGET_INDEX 设置时，此参数在 Python 中会被忽略，但仍保留以备后用）
MAX_VIDEOS=100

# 设备映射（auto/cpu/cuda）
DEVICE_MAP="cuda"
# 每视频最多检测动作数（固定前N个）
MAX_ACTION_NUM=5
# tIoU 阈值（和记忆版保持一致）
TIOU_THRESHOLDS="0.3,0.5"

# ======================== 生成带时间戳的输出文件名 ========================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 修改了文件名，加上 target_index 标识，避免覆盖批量结果
OUTPUT_FILE="/root/autodl-tmp/results/T_BASELINE_video${TARGET_INDEX}_${TIMESTAMP}.json"

# 确保结果目录存在
mkdir -p /root/autodl-tmp/results

# ======================== 运行命令 ========================
echo "===================================================="
echo "🚀 运行 BASELINE 版本（单视频指定测试）"
echo "模型路径: $MODEL_PATH"
echo "视频路径: $VIDEO_DIR"
echo "标注路径: $ANNOTATION_DIR"
echo "数据集划分: $SPLIT"
echo "🎯 指定视频索引: $TARGET_INDEX (第 $((TARGET_INDEX + 1)) 个有效视频)"
echo "tIoU阈值: $TIOU_THRESHOLDS"
echo "输出文件: $OUTPUT_FILE"
echo "===================================================="

# 执行新版Python脚本，新增了 --target_index 参数
python -u /root/autodl-tmp/video_temporal_grounding/test.py \
    --model_path "$MODEL_PATH" \
    --video_dir "$VIDEO_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --device_map "$DEVICE_MAP" \
    --split "$SPLIT" \
    --target_index "$TARGET_INDEX" \
    --max_videos "$MAX_VIDEOS" \
    --output_file "$OUTPUT_FILE" \
    --max_action_num "$MAX_ACTION_NUM" \
    --tiou_thresholds "$TIOU_THRESHOLDS"

# 检查运行状态
if [ $? -eq 0 ]; then
    echo "===================================================="
    echo "✅ 脚本运行完成！结果已保存到: $OUTPUT_FILE"
    echo "===================================================="
else
    echo "===================================================="
    echo "❌ 脚本运行失败！"
    echo "===================================================="
    exit 1
fi