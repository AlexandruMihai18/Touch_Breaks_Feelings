import pandas as pd

def _load_epic_kitchens_data():
    path = '/scratch-shared/dotero/epic_kitchen/annotations/val.json'
    
    df = pd.read_json(path)

    image_paths = [path.replace('/gpfs/home6/Touch_Breaks_Feelings/data', '/scratch-shared') for path in df['image_path'].tolist()]
    audio_series = '/scratch-shared/epic_kitchen/audio/' + df['video_id'] + '.m4a'

    return {
        'image_paths': image_paths,
        'audio_paths': audio_series.tolist(),
        'labels':      (df['type'] == 'touch').astype(int).tolist(), # Vectorized mapping
        'video_ids':   df['video_id'].tolist(),
        'frame_ids':   df['frame_idx'].tolist(),
        'touch_times': df['audio_timestamp_sec'].tolist()
    }

def _load_greatest_hits_data():
    path = '/scratch-shared/dotero/greatest_hits/annotations/val.json'
    
    df = pd.read_json(path)

    
    return {
        'image_paths': df['image_path'],
        'audio_paths': df['audio_path'],
        'labels':      (df['type'] == 'touch').astype(int).tolist(),
        'video_ids':   df['video_id'].tolist(),
        'frame_ids':   df['frame_idx'].tolist(),
        'touch_times': df['timestamp_s'].tolist()  # The specific key for Greatest Hits
    }

def _load_sven_data():
    path = '/scratch-shared/dotero/kubric/annotations/annotations.json'
    
    df = pd.read_json(path)

    image_paths = [path.replace('/gpfs/home6/Touch_Breaks_Feelings/data', '/scratch-shared') for path in df['image_path'].tolist()]
    
    return {
        'image_paths': image_paths,
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
