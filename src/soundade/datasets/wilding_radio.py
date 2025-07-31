from dataclasses import dataclass
from soundade.datasets.base import Dataset

@dataclass
class WildingRadio(Dataset):
    SITE_LEVEL_0: str = "wilding_radio"
    PATTERN: str = (
        "recordings/"
        "(?P<site_level_2>[^/]+)/"
        "(?P<year>\d{4})/"
        "(?P<month>\d{2})/"
        "(?P<day>\d{2})/"
        "(?P<site_level_1>[^-]+)-(?P<timestamp>\d{2}-\d{2}-\d{2}-\d{4})\.[wav|mp3|flac]"
    )
    TIMESTAMP_FORMAT: str = "%H-%d-%m-%Y"
