import json
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# =========================
# Config
# =========================

MODEL_PATH = "/root/autodl-tmp/InternVL3_5-8B"
VIDEO_PATH = "/root/autodl-tmp/wx_camera_1708074606713.mp4"

QUERY = "When does the man drink the beverage in the video?"

WINDOW_SIZE = 1
STRIDE = 1
NUM_FRAMES = 8

INPUT_SIZE = 448
MAX_TILES = 1
CONF_THRESHOLD = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =========================
# Image preprocessing (官方)
# =========================

def build_transform(input_size):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)

        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):

    width, height = image.size
    aspect_ratio = width / height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )

    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))

    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    images = []

    for i in range(blocks):

        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )

        images.append(resized.crop(box))

    if use_thumbnail and blocks != 1:
        images.append(image.resize((image_size, image_size)))

    return images


# =========================
# Video sampling
# =========================

def sample_video(video_path, window_size, stride, num_frames):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    windows = []

    start = 0.0

    while start < duration:

        end = min(start + window_size, duration)

        times = np.linspace(start, end, num_frames)

        frames = []

        for t in times:

            frame_id = min(int(t * fps), total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        windows.append({
            "start": start,
            "end": end,
            "frames": frames
        })

        start += stride

    cap.release()

    return windows, duration


# =========================
# Frames → pixel_values
# =========================

def frames_to_pixel_values(frames):

    transform = build_transform(INPUT_SIZE)

    pixel_values_list = []
    num_patches_list = []

    for frame in frames:

        img = Image.fromarray(frame).convert("RGB")

        tiles = dynamic_preprocess(
            img,
            image_size=INPUT_SIZE,
            use_thumbnail=True,
            max_num=MAX_TILES
        )

        pixels = [transform(tile) for tile in tiles]

        pixels = torch.stack(pixels)

        num_patches_list.append(pixels.shape[0])
        pixel_values_list.append(pixels)

    pixel_values = torch.cat(pixel_values_list)

    return pixel_values, num_patches_list


# =========================
# JSON parser
# =========================

def parse_json(text):

    try:

        start = text.find("{")
        end = text.rfind("}") + 1

        js = text[start:end]

        js = js.replace("'", '"')

        return json.loads(js)

    except:
        return None


# =========================
# Merge segments
# =========================

def merge_segments(segs):

    if not segs:
        return []

    segs = sorted(segs, key=lambda x: x["start"])

    merged = [segs[0]]

    for s in segs[1:]:

        prev = merged[-1]

        if s["start"] <= prev["end"]:

            prev["end"] = max(prev["end"], s["end"])
            prev["confidence"] = max(prev["confidence"], s["confidence"])

        else:

            merged.append(s)

    return merged


# =========================
# Load model
# =========================

print("Loading InternVL3.5...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False
)

print("Model loaded\n")


# =========================
# Sliding window inference
# =========================

windows, duration = sample_video(
    VIDEO_PATH,
    WINDOW_SIZE,
    STRIDE,
    NUM_FRAMES
)

print("Video duration:", duration)
print("Total windows:", len(windows))

results = []

for idx, window in enumerate(windows):

    start = window["start"]
    end = window["end"]

    print("\n====================")
    print(f"Window {idx+1}/{len(windows)}")
    print(f"{start:.2f} - {end:.2f}")

    frames = window["frames"]

    pixel_values, num_patches_list = frames_to_pixel_values(frames)

    pixel_values = pixel_values.to(torch.bfloat16).to(DEVICE)

    video_prefix = "".join(
        [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
    )

    prompt = f"""
{video_prefix}

Segment time range:
{start:.2f} to {end:.2f} seconds.

Question:
{QUERY}

Determine whether the action occurs in THIS segment.

Return ONLY JSON:

{{
"happens": true or false,
"confidence": number between 0 and 1,
"explanation": "short reason"
}}
"""

    gen_cfg = dict(max_new_tokens=256, do_sample=False)

    try:

        response = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            gen_cfg,
            num_patches_list=num_patches_list
        )

        print("Model output:")
        print(response)

        res = parse_json(response)

        if res is None:
            print("JSON parse failed")
            continue

        happens = res.get("happens", False)
        conf = float(res.get("confidence", 0))

        if happens and conf >= CONF_THRESHOLD:

            results.append({
                "start": start,
                "end": end,
                "confidence": conf,
                "explanation": res.get("explanation", "")
            })

    except Exception as e:

        print("Inference error:", e)


# =========================
# Final results
# =========================

merged = merge_segments(results)

print("\n========================")
print("Final detected segments")

if not merged:

    print("No action detected")

else:

    for m in merged:

        print(
            f"{m['start']:.2f}-{m['end']:.2f} "
            f"(conf={m['confidence']:.2f})"
        )

        print("Reason:", m["explanation"])
        print()