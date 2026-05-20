import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from video_utils import sample_video_frames_sliding_window

MODEL_PATH = "/root/autodl-tmp/Qwen3-VL-8B-Instruct"
VIDEO_PATH = "/root/autodl-tmp/wx_camera_1708074606713.mp4"
QUERY = "When does the man drink the beverage in the video?"

WINDOW_SIZE = 2
STRIDE = 2
NUM_FRAMES = 16

CONF_THRESHOLD = 0.7


# =========================================================
# 1. Load model
# =========================================================
print("Loading model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)


# =========================================================
# 2. Sliding window sampling
# =========================================================
print("Sampling video with sliding windows...")

windows, duration = sample_video_frames_sliding_window(
    VIDEO_PATH,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    num_frames=NUM_FRAMES
)

print("Video duration:", duration)
print("Total windows:", len(windows))


# =========================================================
# 3. Iterate windows
# =========================================================
results = []

for idx, window in enumerate(windows):

    frames = window["frames"]
    timestamps = window["timestamps"]
    start_time = window["start"]
    end_time = window["end"]

    print("\n============================")
    print(f"Window {idx}")
    print(f"Time range: {start_time:.2f} - {end_time:.2f}")

    # -----------------------------------------------------
    # Build prompt
    # -----------------------------------------------------

    time_desc = "\n".join(
        [f"Frame {i}: {t:.2f} seconds"
         for i, t in enumerate(timestamps)]
    )

    prompt = f"""
You are a professional video understanding assistant.

The following frames come from a short video segment.

Segment time range:
{start_time:.2f} to {end_time:.2f} seconds.

Frame timestamps:
{time_desc}

Question:
{QUERY}

Task:
Determine whether the action described in the question
happens within THIS video segment.

Decision rules:
- If the action clearly appears, set happens=true
- If the action does NOT appear, set happens=false
- Only judge using the frames shown
- Do NOT assume events outside this segment

Output format:
Return ONLY a JSON object:

{{
 "happens": true or false,
 "confidence": a number between 0 and 1,
 "explanation": "short explanation based on the frames"
}}

Example:
{{"happens": true, "confidence": 0.91, "explanation": "The man lifts a bottle and drinks from it."}}
"""

    # -----------------------------------------------------
    # Build message
    # -----------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img} for img in frames]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True
    )[0]

    print("Model output:", output_text)

    # -----------------------------------------------------
    # Parse result
    # -----------------------------------------------------

    try:
        start = output_text.find("{")
        end = output_text.rfind("}")

        json_str = output_text[start:end+1]

        result = json.loads(json_str)

        happens = result.get("happens", False)
        confidence = result.get("confidence", 0.0)
        explanation = result.get("explanation", "")

        print("Parsed:", happens)
        print("Confidence:", confidence)
        print("Explanation:", explanation)

        if happens and confidence >= CONF_THRESHOLD:
            results.append({
                "start": start_time,
                "end": end_time,
                "confidence": confidence,
                "explanation": explanation
            })

    except Exception as e:
        print("JSON parse failed:", e)


# =========================================================
# 4. Final result
# =========================================================

print("\n============================")
print("Detected segments:")

for r in results:

    print(
        f"{r['start']:.2f} - {r['end']:.2f} "
        f"(confidence={r['confidence']:.2f})"
    )

    print("Reason:", r["explanation"])
    print()