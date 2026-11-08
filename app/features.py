"""특징값 — 학습과 예측(/predict)이 같은 함수를 쓴다.

입력 컬럼: route_id, station_id, t(기준 시각), pred_sec(도착까지 남은 초), seat_now, plate_no,
          ahead_sec·ahead_seat(앞차의 남은 초·좌석, add_ahead로 붙인다)
"""
import numpy as np
import pandas as pd

from app import config

FEATURES = ["route_id", "station_id", "seat_now", "pred_sec", "tod_min", "dow", "weekend", "capacity",
            "gap_ahead", "seat_ahead"]


def add_ahead(df: pd.DataFrame) -> pd.DataFrame:
    """같은 요청(시각·노선·정류장)에서 바로 앞에 오는 버스의 남은 초·좌석. 맨 앞 버스는 NaN.
    /predict 요청의 arrivals 목록과 같은 단위로 계산한다."""
    df = df.sort_values(["t", "route_id", "station_id", "pred_sec"])
    g = df.groupby(["t", "route_id", "station_id"], sort=False)
    return df.assign(ahead_sec=g.pred_sec.shift(), ahead_seat=g.seat_now.shift())


def make(df: pd.DataFrame, capacity: dict, stations=None) -> pd.DataFrame:
    """stations: 정류장 카테고리 목록. 없으면 수집 정류장 5곳 (전체 정류장 모델은 meta의 목록을 넘긴다)."""
    x = pd.DataFrame(index=df.index)
    # 카테고리 목록을 고정해야 학습·예측의 코드값이 같다
    x["route_id"] = pd.Categorical(df.route_id, categories=list(config.ROUTES))
    x["station_id"] = pd.Categorical(df.station_id, categories=list(stations or config.STATIONS))
    x["seat_now"] = df.seat_now.where(df.seat_now >= 0).astype(float)
    x["pred_sec"] = df.pred_sec.astype(float)
    x["tod_min"] = df.t.dt.hour * 60 + df.t.dt.minute
    x["dow"] = df.t.dt.day_of_week
    x["weekend"] = (x.dow >= 5).astype(int)
    x["capacity"] = df.plate_no.map(capacity).astype(float)   # 처음 보는 차량은 NaN
    x["gap_ahead"] = (df.pred_sec - df.ahead_sec).astype(float)   # 앞차와의 간격(초)
    x["seat_ahead"] = df.ahead_seat.where(df.ahead_seat >= 0).astype(float)
    return x[FEATURES]


def seat_stage(seats) -> np.ndarray:
    """명세 좌석 단계: 5석 이상 OK / 1~4석 CAUTION / 0석 RISK."""
    s = np.asarray(seats, dtype=float)
    return np.where(s >= 5, "OK", np.where(s >= 1, "CAUTION", "RISK"))
