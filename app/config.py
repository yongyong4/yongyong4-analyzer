"""대상 노선·정류장, 수집 시간대, 환경변수."""
import os
from datetime import datetime, time

from dotenv import load_dotenv

load_dotenv()   # 실행 폴더의 .env (git 제외)

COLLECTOR_BASE_URL = os.getenv("COLLECTOR_BASE_URL", "")
DATA_FROM = os.getenv("DATA_FROM", "2026-10-28")
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# 서비스 DB (bus_pass, model_eval). MYSQL_HOST가 비어 있으면 DB에 쓰지 않는다
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "yongyong")
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")

# 수집 광역 노선 (gbisRouteId → 노선번호)
ROUTES = {
    "228000174": "5000",
    "228000429": "5001A",
    "228000431": "5003A",
    "228000175": "5005",
    "228000447": "5005(거점)",
}

# 수집 정류장 (stationId → 이름)
STATIONS = {
    "228000745": "시청.용인대역",
    "228000172": "동백역",
    "228000684": "강남대역",
    "228000682": "기흥역",
    "228000680": "상미마을",
}

# 수집 시간대 = 예측 대상 시간대 (밖이면 400)
WEEKDAY_WINDOW = (time(6, 0), time(10, 30))
WEEKEND_WINDOW = (time(10, 0), time(15, 0))


def in_window(at: datetime) -> bool:
    start, end = WEEKEND_WINDOW if at.weekday() >= 5 else WEEKDAY_WINDOW
    return start <= at.time() <= end
