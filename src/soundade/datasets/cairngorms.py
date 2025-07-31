from dataclasses import dataclass
from soundade.datasets.base import Dataset

@dataclass
class Cairngorms(Dataset):
    SITE_LEVEL_0: str = "cairngorms"
    PATTERN: str = (
        "(?<recorder_model>[^/]+)_(?P<site_level_1>[^/]+)/"
        "(?P<date>\d{8})/"
        "(?P<timestamp>\d{8}_\d{6})\.[wav|flac|mp3]"
    )
    TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
