import argparse
import librosa
import os
import pandas as pd
import torch

from PIL import Image
from qwen_baseline_utils import load_data
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

AUDIO_MODEL_ID = 'Qwen/Qwen2-Audio-7B-Instruct'
AUDIO_PROMPT = """
You are an expert acoustic-to-visual translator. Analyze the non-speech audio provided and generate a descriptive paragraph that a vision model can use to match this audio with an image.

Follow these strict constraints:
1. Focus Area: Prioritize physical interactions, knocks, impacts, collisions, and mechanical friction. 
2. Transmute Sound to Physics: Describe the implied physical properties of the objects involved—their estimated mass (heavy/light), material texture (wood, metal, plastic, hollow, solid), and the force of the interaction.
3. Spatial Context: Detail the acoustic space (e.g., tight indoor room, large echoing hall, open outdoor area) based on the reverberation and decay of the impacts.
4. Output Format: Write exactly one continuous, highly descriptive paragraph. Do not use bullet points, markdown bolding, lists, or introductory phrases like "The audio features...". 
5. Length: Keep the final paragraph strictly between 100 and 150 words.

Begin the description immediately.
"""

VISUAL_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
VISUAL_PROMPT_WITH_AUDIO = """
You are given:
1. An image
2. A description of the related audio context: {audio_description}

Task:
Determine whether the image contains a clear physical touch or contact event between two objects, two people, or a person and an object. 

Examples of touch/contact:
- Holding, grabbing, pushing, hugging
- Hand touching an object
- Physical interaction with surfaces or tools
- Collision or impact

Output rules:
- Return only:
    1 = physical touch/contact is present
    0 = no physical touch/contact is visible
- Do not explain your answer.
- If the contact is ambiguous or unclear, return 0.
"""

VISUAL_PROMPT_POINT_OF_TOUCH = """
You are given an image.
Task:
Identify the point of physical touch or contact in the image, if present.
Output rules:
- If physical touch/contact is present, return the coordinates of the point of contact in the format
    (x, y) where x and y are the pixel coordinates of the contact point in the image.
- If no physical touch/contact is visible, return (0, 0).
- If the contact is ambiguous or unclear, return (0, 0).
"""

def generate_description_from_audio(
    audio_path: str,
    processor: AutoProcessor,
    model: Qwen2AudioForConditionalGeneration
):
    audio_description_path = audio_path.replace('.wav', '_description.txt')

    if os.path.exists(audio_description_path):
        return audio_description_path
    
    audio, _ = librosa.load(audio_path, sr=16000)

    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": AUDIO_PROMPT}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios = [audio]

    inputs = processor(text=text, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    generate_ids = model.generate(**inputs, max_new_tokens=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    with audio_description_path.open('w', encoding='utf-8') as f:
        f.write(response)

    return audio_description_path

def process_image_with_audio_description(
    image_path: str, 
    audio_description_path: str,
    processor : AutoProcessor = None,
    model : Qwen2_5_VLForConditionalGeneration = None
    ) -> None:
    '''
    Uses the Qwen2.5-VL model to process the image together with the audio description, and prints the model's response.
    '''

    with open(audio_description_path, 'r', encoding='utf-8') as f:
        audio_description = f.read()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": VISUAL_PROMPT_WITH_AUDIO.format(audio_description=audio_description)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"Image path: {image_path}\nAudio_Description: {audio_description_path.read_text()}\nResponse: {response}\n")
    label = 1 if response.strip() == "1" else 0
    return label

def identify_point_of_touch(image_path, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": VISUAL_PROMPT_POINT_OF_TOUCH},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"Image path: {image_path}\nResponse: {response}\n")

    try:
        x, y = map(int, response.strip().strip("()").split(","))
        return (x, y)
    except ValueError:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-to-Visual Description Baseline")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset', choices=['EpicKitchen', 'GreatestHits'])
    args = parser.parse_args()

    data = load_data(args.dataset)

    # Load the audio model
    audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_ID)
    audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(AUDIO_MODEL_ID).cuda()

    # Load the visual model
    visual_processor = AutoProcessor.from_pretrained(VISUAL_MODEL_ID)
    visual_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VISUAL_MODEL_ID).cuda()

    audio_paths = data['audio_paths']
    image_paths = data['image_paths']
    labels = data['labels']

    audio_description_paths = []

    for audio_path in audio_paths:
        audio_description_path = generate_description_from_audio(audio_path, audio_processor, audio_model)
        print(f"Generated description for {audio_path} at {audio_description_path}")
        audio_description_paths.append(audio_description_path)
    
    predictions = []
    points_of_touch = []

    for image_path, audio_description_path, label in zip(image_paths, audio_description_paths, labels):
        prediction = process_image_with_audio_description(image_path, audio_description_path, visual_processor, visual_model)
        point_of_touch = identify_point_of_touch(image_path, visual_processor, visual_model)

        if prediction == 0 and label == 0:
            point_of_touch = None
        
        predictions.append(prediction)
        points_of_touch.append(point_of_touch)

        print(f"Final Prediction for {image_path}: Touch Detected = {prediction}, Point of Touch = {point_of_touch}\n")
    
    results = pd.DataFrame({
        'frame_id': data['frame_ids'],
        'video_id': data['video_ids'],
        'frame_path': data['image_paths'],
        'audio_path': data['audio_paths'],
        'label': data['labels'],
        'prediction': predictions,
        'x_touch': [pt[0] if pt else None for pt in points_of_touch],
        'y_touch': [pt[1] if pt else None for pt in points_of_touch],
    })

    results.to_csv(f'results/{args.dataset}_qwen2.5_visual_audio_description_baseline_predictions.csv', index=False)
    