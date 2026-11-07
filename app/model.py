"""모델 적용·저장·불러오기 — 학습(train)과 예측 서버가 같이 쓴다."""
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from app import config, features


def apply(seat_model, arrival_model, df: pd.DataFrame, capacity: dict):
    """좌석 = 현재 좌석 - 예측 감소량 (0~현재 좌석), 도착 = GBIS 도착 예정 + 예측 오차(분)."""
    x = features.make(df, capacity)
    drop = seat_model.predict(x)
    seat_now = x.seat_now.to_numpy()
    seats = np.clip(np.round(seat_now - drop), 0, seat_now)          # 현재 좌석이 없으면 NaN
    pred_at = df.t + pd.to_timedelta(df.pred_sec, unit="s")
    arrival_at = pred_at + pd.to_timedelta(arrival_model.predict(x), unit="m")
    return seats, arrival_at


def save(version: str, seat_model, arrival_model, meta: dict) -> str:
    path = os.path.join(config.MODEL_DIR, version)
    os.makedirs(path, exist_ok=True)
    seat_model.booster_.save_model(os.path.join(path, "seat.txt"))
    arrival_model.booster_.save_model(os.path.join(path, "arrival.txt"))
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return path


def active() -> dict:
    """모델 종류별 사용 중 버전과 평일·주말 사용 여부.
    예: {"seat": {"version": "v3-20261110", "weekday": true, "weekend": true},
         "arrival": {"version": "v3-20261110", "weekday": true, "weekend": false}}
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
    path = os.path.join(config.MODEL_DIR, version)
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    seat = lgb.Booster(model_file=os.path.join(path, "seat.txt"))
    arrival = lgb.Booster(model_file=os.path.join(path, "arrival.txt"))
    return seat, arrival, meta
