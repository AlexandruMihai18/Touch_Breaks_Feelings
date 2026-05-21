import json

def _load_epic_kitchens_data():
    EPIC_KITCHEN_ANNOTATIONS_PATH = './data/epic_kitchen/annotations/annotations.json'
    data = dict(
        image_paths=[],
        audio_paths=[],
        labels=[],
        video_ids=[],
        frame_ids=[],
        touch_times=[]
    )

    with open(EPIC_KITCHEN_ANNOTATIONS_PATH, 'r') as f:
        uncurated_data = json.load(f)
        for image_data in uncurated_data:
            data['image_paths'].append(image_data['image_path'])
            data['audio_paths'].append(image_data['audio_path'])
            data['labels'].append(1 if image_data['type'] == 'touch' else 0)
            data['video_ids'].append(image_data['video_id'])
            data['frame_ids'].append(image_data['frame_idx'])
            data['touch_times'].append(image_data['audio_timestamp_sec'])

    return data

def _load_greatest_hits_data():
    GREATEST_HITS_ANNOTATIONS_PATH = './data/greatest_hits/annotations/annotations.json'
    data = dict(
        image_paths=[],
        audio_paths=[],
        labels=[],
        video_ids=[],
        frame_ids=[],
        touch_times=[]
    )

    with open(GREATEST_HITS_ANNOTATIONS_PATH, 'r') as f:
        uncurated_data = json.load(f)
        for image_data in uncurated_data:
            data['image_paths'].append(image_data['image_path'])
            data['audio_paths'].append(image_data['audio_path'])
            data['labels'].append(1 if image_data['type'] == 'touch' else 0)
            data['video_ids'].append(image_data['video_id'])
            data['frame_ids'].append(image_data['frame_idx'])
            data['touch_times'].append(image_data['timestamp_s'])

    return data

def _load_sven_data():
    SVEN_ANNOTATIONS_PATH = './data/sven/annotations/annotations.json'
    data = dict(
        image_paths=[],
        labels=[],
        video_ids=[],
        frame_ids=[],
        touch_times=[]
    )

    with open(SVEN_ANNOTATIONS_PATH, 'r') as f:
        uncurated_data = json.load(f)
        for image_data in uncurated_data:
            data['image_paths'].append(image_data['image_path'])
            data['labels'].append(1 if image_data['type'] == 'touch' else 0)
            data['video_ids'].append(image_data['video_id'])
            data['frame_ids'].append(image_data['frame_idx'])
            data['touch_times'].append(image_data['timestamp_s'])

    return data

def load_data(dataset_name: str):
    if dataset_name == "EpicKitchens":
        data = _load_epic_kitchens_data()
    elif dataset_name == "GreatestHits":
        data = _load_greatest_hits_data()
    elif dataset_name == "Sven":
        data = _load_sven_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return data
