import argparse
import pandas as pd
import torch

from PIL import Image
from qwen_baseline_utils import load_data
from transformers import AutoModelForCausalLM, AutoProcessor

VISUAL_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
VISUAL_PROMPT_WITHOUT_AUDIO = """
You are given an image.
    
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

def process_image_without_audio(image_path, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": VISUAL_PROMPT_WITHOUT_AUDIO},
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset', choices=['EpicKitchen', 'GreatestHits', 'Sven'])
    args = parser.parse_args()

    data = load_data(args.dataset)

    processor = AutoProcessor.from_pretrained(VISUAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(VISUAL_MODEL_ID)

    predictions = []
    points_of_touch = []

    image_paths = data['frame_paths'].tolist()

    for image_path in image_paths:
        prediction = process_image_without_audio(image_path, processor, model)
        point_of_touch = identify_point_of_touch(image_path, processor, model)

        if prediction == 0:
            point_of_touch = None

        predictions.append(prediction)
        points_of_touch.append(point_of_touch)
        
        print(f"Image: {image_path}, Touch Prediction: {prediction}, Point of Touch: {point_of_touch}\n")
    
    results = pd.DataFrame({
        'frame_id': data['frame_ids'],
        'video_id': data['video_ids'],
        'frame_path': data['frame_paths'],
        'audio_path': data['audio_paths'],
        'label': data['labels'],
        'prediction': predictions,
        'x_touch': [pt[0] for pt in points_of_touch],
        'y_touch': [pt[1] for pt in points_of_touch],
    })

    results.to_csv(f'{args.dataset}_qwen2.5_visual_baseline_predictions.csv', index=False)

