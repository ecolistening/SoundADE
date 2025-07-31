from dataclasses import dataclass
from soundade.datasets.base import Dataset

@dataclass
class NatureSense(Dataset):
    PATTERN = (
        "(?P<recorder_model>Audiomoths|Song_Meter_Mini_2|SM4_Bat)/"
        "(?P<site_level_1>[^/]+)/"
        "[^/_]+_(?P<site_level_2>[^/_]+(?:[ _ ][^/_]+)*?)(?:_\d{8})?/"
        "(?:[^_/]+_)?"
        "(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
        "(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        "\.(?:wav|WAV|flac|FLAC|mp3|MP3)"
    )
