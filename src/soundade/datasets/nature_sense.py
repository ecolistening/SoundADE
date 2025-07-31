from dataclasses import dataclass
from soundade.datasets.base import Dataset

@dataclass
class NatureSense(Dataset):
    SITE_LEVEL_0: str = "nature_sense"
    SMM2_PATTERN = (
        "(?P<recorder_model>Song_Meter_Mini_2)/"
        "(?P<site_level_1>[^/]+)/"
        "[^/]+_(?P<site_level_2>[^/]+)/"
        ".*?/"
        "[^_]+_(?P<timestamp>\d{8}_\d{6})\.[wav|mp3|flac]"
    )
    AUDIO_MOTH_PATTERN = (
        "(?P<recorder_model>Audiomoth)s/"
        "(?P<site_level_1>[^/]+)/"
        "(?P<site_level_2>[^/]+_[^/]+)_\d+/"
        "(?P<timestamp>\d{8}_\d{6})\.[wav|mp4|flac]"
    )
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
