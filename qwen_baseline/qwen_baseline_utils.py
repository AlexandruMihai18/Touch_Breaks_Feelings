import json
import pandas as pd

def _load_epic_kitchens_data():
    path = './data/epic_kitchen/annotations/val.json'
    
    df = pd.read_json(path)
    
    return {
        'image_paths': df['image_path'].tolist(),
        'audio_paths': df['audio_path'].tolist(),
        'labels':      (df['type'] == 'touch').astype(int).tolist(), # Vectorized mapping
        'video_ids':   df['video_id'].tolist(),
        'frame_ids':   df['frame_idx'].tolist(),
        'touch_times': df['audio_timestamp_sec'].tolist()
    }

def _load_greatest_hits_data():
    path = './data/greatest_hits/annotations/annotations.json'
    
    df = pd.read_json(path)
    
    return {
        'image_paths': df['image_path'].tolist(),
        'audio_paths': df['audio_path'].tolist(),
        'labels':      (df['type'] == 'touch').astype(int).tolist(),
        'video_ids':   df['video_id'].tolist(),
        'frame_ids':   df['frame_idx'].tolist(),
        'touch_times': df['timestamp_s'].tolist()  # The specific key for Greatest Hits
    }

def _load_sven_data():
    path = './data/sven/annotations/annotations.json'
    
    df = pd.read_json(path)
    
    return {
        'image_paths': df['image_path'].tolist(),
        'labels':      (df['type'] == 'touch').astype(int).tolist(),
        'video_ids':   df['video_id'].tolist(),
        'frame_ids':   df['frame_idx'].tolist(),
        'touch_times': df['timestamp_s'].tolist()  # The specific key for Sven
    }

def load_data(dataset_name: str):
    if dataset_name == "EpicKitchen":
        data = _load_epic_kitchens_data()
    elif dataset_name == "GreatestHits":
        data = _load_greatest_hits_data()
    elif dataset_name == "Sven":
        data = _load_sven_data()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return data
