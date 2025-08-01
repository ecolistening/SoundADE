from dataclasses import dataclass
from soundade.datasets.base import Dataset

@dataclass
class NatureSense(Dataset):
    PATTERN = (
        r"(?P<recorder_model>Audiomoths|Song_Meter_Mini_2|SM4_Bat)/"
        r"(?P<site_level_1>[^/]+)/"
        r"(?P<site_level_2>[^/]+)/"
        r"(?:\d{8}/)?"
        r"(?P<device_id>[^/_]+)_"
        r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
        r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        r"\.(?i:wav|flac|mp3)"
    )
