"""수집 서버에서 학습용 CSV 받기 (/export/zip)."""
import io
import os
import urllib.request
import zipfile
from datetime import date

from app import config

TABLES = "bus_location,bus_arrival,route_station"


def download(date_from: str = config.DATA_FROM, date_to: str | None = None) -> str:
    """기간 CSV를 data/raw/ 에 풀고 그 경로를 돌려준다."""
    date_to = date_to or date.today().isoformat()
    url = (f"{config.COLLECTOR_BASE_URL}/export/zip"
           f"?from={date_from}&to={date_to}&tables={TABLES}")
    raw_dir = os.path.join(config.DATA_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as res:
        data = res.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith(".csv") and "/" not in name and not name.startswith("."):
                z.extract(name, raw_dir)
    return raw_dir
