"""모델 적용·저장·불러오기 — 학습(train)과 예측 서버가 같이 쓴다."""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from app import config, features


def seat_upper(x: pd.DataFrame, rise: bool) -> np.ndarray:
    """예측 좌석의 최대값. 보통은 현재 좌석(타기만 하는 구간),
    rise면 정원까지 (5곳 밖 정류장 = 서울 방향 내리는 구간 포함). 정원을 모르면 현재 좌석."""
    seat_now = x.seat_now.to_numpy()
    return np.fmax(seat_now, x.capacity.to_numpy()) if rise else seat_now


def predict_seats(seat_model, df: pd.DataFrame, capacity: dict, stations=None) -> np.ndarray:
    """좌석 = 현재 좌석 - 예측 감소량 (0~최대값). 현재 좌석이 없으면 NaN.
    stations를 넘기면 5곳 밖 정류장 모델(seat_all): 정류장 목록이 다르고 좌석이 늘 수 있다."""
    x = features.make(df, capacity, stations)
    return np.clip(np.round(x.seat_now.to_numpy() - seat_model.predict(x)), 0, seat_upper(x, stations is not None))


def predict_arrival(arrival_model, df: pd.DataFrame, capacity: dict) -> pd.Series:
    """도착 = GBIS 도착 예정 + 예측 오차(분)."""
    x = features.make(df, capacity)
    pred_at = df.t + pd.to_timedelta(df.pred_sec, unit="s")
    return pred_at + pd.to_timedelta(arrival_model.predict(x), unit="m")


def apply(seat_model, arrival_model, df: pd.DataFrame, capacity: dict):
    return predict_seats(seat_model, df, capacity), predict_arrival(arrival_model, df, capacity)


def save(version: str, models: dict, meta: dict) -> str:
    """models: {"seat": ..., "arrival": ..., "seat_all": ...} → 버전 폴더에 종류별 파일 1개씩."""
    path = os.path.join(config.MODEL_DIR, version)
    os.makedirs(path, exist_ok=True)
    for kind, m in models.items():
        m.booster_.save_model(os.path.join(path, f"{kind}.txt"))
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return path


def active() -> dict:
    """모델 종류별 사용 중 버전과 평일·주말 사용 여부.
    예: {"seat": {"version": "v3-20261110", "weekday": true, "weekend": true},
         "arrival": {"version": "v3-20261110", "weekday": true, "weekend": false},
         "seat_all": {"version": "v3-20261110", "weekday": true, "weekend": true}}   # 5곳 밖 정류장 좌석
    """
    path = os.path.join(config.MODEL_DIR, "active.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_active(state: dict) -> None:
    with open(os.path.join(config.MODEL_DIR, "active.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load(version: str):
    """(종류별 모델, meta). 이전 버전에는 seat_all이 없을 수 있다."""
    path = os.path.join(config.MODEL_DIR, version)
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    boosters = {kind: lgb.Booster(model_file=os.path.join(path, f"{kind}.txt"))
                for kind in ("seat", "arrival", "seat_all") if os.path.exists(os.path.join(path, f"{kind}.txt"))}
    return boosters, meta
