#!/bin/bash
# Charades视频动作定位运行脚本（增强记忆版｜单视频测试）
# 使用说明：执行 chmod +x runsingle_memory.sh && ./runsingle_memory.sh 或者直接 bash runsingle_memory.sh

# ======================== 配置参数 ========================
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

# 批量处理的最大视频数（单视频模式下此参数被忽略）
MAX_VIDEOS=100

# 设备映射（auto/cpu/cuda）
DEVICE_MAP="cuda"
# 置信度阈值
CONF_THRESHOLD=0.7
# 每视频最多检测动作数
MAX_ACTION_NUM=5
# tIoU评估阈值
TIOU_THRESHOLDS="0.3,0.5"

# ======================== 生成带时间戳的输出文件名 ========================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 输出文件名带上 target_index，避免和批量运行的结果混淆
OUTPUT_FILE="/root/autodl-tmp/results/TM_video${TARGET_INDEX}_${TIMESTAMP}.json"

# 确保结果目录存在
mkdir -p /root/autodl-tmp/results

# ======================== 运行命令 ========================
echo "===================================================="
echo "🚀 开始运行Charades视频动作定位（非对称记忆增强版｜单视频指定测试）"
echo "模型路径: $MODEL_PATH"
echo "视频路径: $VIDEO_DIR"
echo "🎯 指定视频索引: $TARGET_INDEX (第 $((TARGET_INDEX + 1)) 个有效视频)"
echo "输出文件: $OUTPUT_FILE"
echo "===================================================="

# 执行Python脚本 (新增了 --target_index 参数)
python -u /root/autodl-tmp/video_temporal_grounding/testmemorysingle.py \
    --model_path "$MODEL_PATH" \
    --video_dir "$VIDEO_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --device_map "$DEVICE_MAP" \
    --split "$SPLIT" \
    --target_index "$TARGET_INDEX" \
    --max_videos "$MAX_VIDEOS" \
    --output_file "$OUTPUT_FILE" \
    --conf_threshold "$CONF_THRESHOLD" \
    --max_action_num "$MAX_ACTION_NUM" \
    --tiou_thresholds "$TIOU_THRESHOLDS"

# 修复状态检查逻辑
if [[ $? -eq 0 ]]; then
    echo "===================================================="
    echo "✅ 脚本运行完成！结果已保存到: $OUTPUT_FILE"
    echo "===================================================="
else
    echo "===================================================="
    echo "❌ 脚本运行失败！"
    echo "===================================================="
    exit 1
fi