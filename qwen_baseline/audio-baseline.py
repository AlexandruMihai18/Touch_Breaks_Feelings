import argparse
import librosa

from qwen_baseline_utils import load_data
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

AUDIO_MODEL_ID = 'Qwen/Qwen2-Audio-7B-Instruct'
AUDIO_PROMPT = """
User: Analyze the provided audio file and detect all instances of physical contact, touch, or impact (human-to-human, human-to-object, or object-to-object collisions). 

Identify the exact onset timestamp (in seconds) for every discrete touch event you detect.

You must follow this format strictly:
Response: [t0, t1, ..., tn]

Rules:
- Replace t0, t1, etc., with the numerical timestamp in seconds (e.g., 0.45, 1.20, 3.85).
- Arrange the timestamps chronologically.
- Do not include time ranges or durations—only the exact onset second of the touch.
- Do not include any introductory text, explanations, markdown formatting, or closing remarks. Output ONLY the requested format.

Example of a valid output:
Response: [0.3245, 1.1567, 2.4089, 5.8101]
"""

def predict_touch_times_for_audio(
    audio_path: str,
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration
):
    predicted_touch_times = []

    audio, _ = librosa.load(audio_path, sr=16000)
    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": str(audio_path)}, # Convert Path to str just in case
            {"type": "text", "text": AUDIO_PROMPT}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios = [audio]

    inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    input_len = inputs.input_ids.shape[1]

    generate_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_tokens = generate_ids[0][input_len:]

    response = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    print(f"Model response: {response}")

    # 4. Parse the array directly (Notice it outputs just the array now based on your debug log)
    if response:
        try:
            if response.startswith("Response:"):
                response = response[len("Response:"):].strip()
            predicted_touch_times = [float(t.strip()) for t in response.strip("[]").split(",") if t.strip()]
        except ValueError:
            print(f"Failed to parse floats from response: {response}")
            return []

    return predicted_touch_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-based touch detection baseline")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset", choices=["EpicKitchens", "GreatestHits"])

    args = parser.parse_args()

    data = load_data(args.dataset)

    processor = AutoProcessor.from_pretrained(AUDIO_MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(AUDIO_MODEL_ID).cuda()

    audio_paths = data["audio_path"].tolist()
    video_ids = data["video_id"].tolist()

    ### TODO TODO
