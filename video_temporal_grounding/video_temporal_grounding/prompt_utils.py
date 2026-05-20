def build_temporal_prompt(timestamps, query, duration):
    prompt = (
        "You are given a video sampled at different timestamps.\n"
        f"The video duration is {duration:.2f} seconds.\n\n"
    )

    for i, t in enumerate(timestamps):
        prompt += f"Frame {i}: time={t:.2f} seconds.\n"

    prompt += f"""
Question:
"{query}"

Rules:
- The answer must be within the video duration
- Use the timestamps above
- Be precise

Output JSON only:
{{"start": number, "end": number}}
"""
    return prompt