def _load_epic_kitchens_data():
    pass

def _load_greatest_hits_data():
    pass

def _load_sven_data():
    pass

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
