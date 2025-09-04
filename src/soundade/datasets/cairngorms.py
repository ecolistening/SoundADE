from dataclasses import dataclass
from soundade.datasets.base import Dataset

@dataclass
class Cairngorms(Dataset):
    PATTERN: str = (
        "(?P<recorder_model>[^/]+)/"
        "(?P<site_level_1>[^/]+)/"
        "\d{4}/"
        "\d{2}/"
        "(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
        "(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        "\.[wav|flac|mp3]"
    )
