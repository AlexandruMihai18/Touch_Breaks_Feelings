import argparse
import librosa
import pandas as pd

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

TOLERANCE = 0.1

def generate_predicted_labels(predicted_touch_times, touch_times):
    predicted_labels = []
    
    for pred_times, true_times in zip(predicted_touch_times, touch_times):
        for true_time in true_times:
            if any(abs(pred_time - true_time) <= TOLERANCE for pred_time in pred_times):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
    
    return predicted_labels

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
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset", choices=["EpicKitchen", "GreatestHits"])

    args = parser.parse_args()

    data = load_data(args.dataset)

    processor = AutoProcessor.from_pretrained(AUDIO_MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(AUDIO_MODEL_ID).cuda()

    audio_paths = data["audio_paths"]
    video_ids = data["video_ids"]
    touch_times = data["touch_times"]

    predicted_touch_times = []

    for audio_path in audio_paths:
        predicted_times = predict_touch_times_for_audio(audio_path, processor, model)
        predicted_touch_times.append(predicted_times)

    predicted_labels = generate_predicted_labels(predicted_touch_times, touch_times)

    results = pd.DataFrame({
        'frame_id': data['frame_ids'],
        'video_id': data['video_ids'],
        'frame_path': data['image_paths'],
        'audio_path': data['audio_paths'],
        'label': data['labels'],
        'prediction': predicted_labels,
        'x_touch': [None] * len(predicted_labels),
        'y_touch': [None] * len(predicted_labels),
    })

