import re

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

    def _get_match(self, file_path):
        file_path = audio_dict["local_file_path"]
        audio_moth_match = re.search(self.AUDIO_MOTH_PATTERN, file_path, flags=re.IGNORECASE)
        song_meter_match = re.search(self.SMM2_PATTERN, file_path, flags=re.IGNORECASE)
        return audio_moth_match if audio_moth_match else (song_meter_match if song_meter_match else None)
