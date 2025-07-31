import logging
import pandas as pd
import re

from dataclasses import dataclass
from pathlib import Path

from soundade.datasets.base import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class Kilpisjarvi(Dataset):
    SITE_LEVEL_0: str = "kilpisjarvi"
    PATTERN = (
        "(?P<site_level_1>[^/]+)/"
        "Data/"
        "(?P<site_level_2>SMA\d{5})_"
        "(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
        "(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        "\.[wav|flac|mp3]"
    )
    SMM_SUMMARY = "(?P<recorder>SMA\d{5})_Summary.txt"

    def index_sites(self, root_dir: str | Path) -> pd.DataFrame:
        site_data = []
        hemisphere_to_sign = {"N": 1, "S": -1, "E": 1, "W": -1}
        for file_path in root_dir.rglob(self.SMM_SUMMARY):
            try:
                site_level_1, site_level_2 = file_path.parent.name, match.group("recorder")
                match = re.search(self.SMM_SUMMARY, file_path.name)
                row = next(pd.read_csv(file_path, chunksize=1)).iloc[0]
                latitude, lat_hemi, longitude, lon_hemi = row[2:6]
                site_data.append({
                    "site_id": str(uuid.uuid4()),
                    "site_name": "/".join([site_level_1, site_level_2]),
                    "location": site_level_1,
                    "recorder": match.group("recorder"),
                    "latitude": float(latitude) * hemisphere_to_sign[str(lat_hemi).strip()],
                    "longitude": float(longitude) * hemisphere_to_sign[str(lon_hemi).strip()],
                    "country": "Finland",
                    "timezone": "Europe/Helsinki",
                })
            except Exception as e:
                log.warning(e)
                continue
        return pd.DataFrame(site_data)
