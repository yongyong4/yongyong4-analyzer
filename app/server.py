"""POST /predict — 명세 「도착·좌석 예측」.

사용 중 모델(models/active.json)을 불러와 예측한다. active.json이 바뀌면(매일 학습) 다음 요청 때 다시 불러온다.
학습 판정에 떨어진 칸: 도착은 받은 값 그대로, 좌석은 null (백엔드가 OBSERVED로 표시).
실행: .venv/bin/uvicorn app.server:app --port 8000
"""
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import config, features, model

KST = ZoneInfo("Asia/Seoul")
KINDS = ("seat", "arrival")

class Arrival(BaseModel):
    plateNo: str
    predictedBusArrivalAt: datetime
    remainSeats: int | None = None


class PredictRequest(BaseModel):
    gbisRouteId: str
    boardStationId: str
    requestAt: datetime
    arrivals: list[Arrival]


class Prediction(BaseModel):
    plateNo: str
    predictedBusArrivalAt: datetime
    predictedRemainSeats: int | None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    modelVersion: str


class BadRequest(Exception):
    pass


class ModelError(Exception):
    pass


class Models:
    """active.json 기준 종류별 모델. 파일 수정 시각이 바뀌면 다시 불러온다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.mtime = None
        self.state, self.boosters, self.capacity = {}, {}, {}

    def get(self):
        path = os.path.join(config.MODEL_DIR, "active.json")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            raise ModelError("사용 중인 모델이 없어요 (python -m app.train 먼저 실행)")
        with self.lock:
            if mtime != self.mtime:
                try:
                    state = model.active()
                    boosters, capacity = {}, {}
                    for kind in KINDS:
                        seat, arrival, meta = model.load(state[kind]["version"])
                        boosters[kind] = seat if kind == "seat" else arrival
                        capacity[kind] = meta["capacity"]
                except Exception as e:
                    raise ModelError(f"모델을 불러오지 못했어요: {e}")
                self.state, self.boosters, self.capacity, self.mtime = state, boosters, capacity, mtime
            return self.state, self.boosters, self.capacity

    def version(self) -> str:
        v = {kind: self.state[kind]["version"] for kind in KINDS}
        return v["seat"] if v["seat"] == v["arrival"] else f"seat:{v['seat']},arrival:{v['arrival']}"


models = Models()


def kst(dt: datetime) -> datetime:
    return dt.replace(tzinfo=KST) if dt.tzinfo is None else dt.astimezone(KST)


def naive(dt: datetime) -> datetime:
    return kst(dt).replace(tzinfo=None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        models.get()   # 켜질 때 한 번 불러 둔다
    except ModelError as e:
        print(f"[경고] {e}")   # 서버는 켜 두고, 요청이 오면 500
    yield


app = FastAPI(title="Yongyong Analyzer", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, e: RequestValidationError):
    # 명세: 필수값 누락은 400 (FastAPI 기본은 422)
    return JSONResponse(status_code=400, content={"message": "필수값이 없거나 형식이 맞지 않아요", "detail": str(e.errors())})


@app.exception_handler(BadRequest)
async def bad_request(_: Request, e: BadRequest):
    return JSONResponse(status_code=400, content={"message": str(e)})


@app.exception_handler(ModelError)
async def model_error(_: Request, e: ModelError):
    # 명세: 모델 로드나 예측 실패는 500 → 백엔드는 받은 값을 그대로 쓴다 (OBSERVED)
    return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/health")
def health():
    try:
        models.get()
        return {"status": "UP", "modelVersion": models.version(), "active": models.state}
    except ModelError as e:
        return JSONResponse(status_code=500, content={"status": "DOWN", "message": str(e)})


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.gbisRouteId not in config.ROUTES:
        raise BadRequest("수집 광역 노선이 아니에요")
    if req.boardStationId not in config.STATIONS:
        raise BadRequest("수집 정류장이 아니에요")
    at = naive(req.requestAt)
    if not config.in_window(at):
        raise BadRequest("수집 시간대가 아니에요")

    state, boosters, capacity = models.get()
    if not req.arrivals:
        return PredictResponse(predictions=[], modelVersion=models.version())

    # 학습 데이터와 같은 모양으로: 기준 시각 t, 남은 초, 현재 좌석(-1 = 없음), 앞차 정보
    df = pd.DataFrame({
        "t": pd.Timestamp(at),
        "route_id": req.gbisRouteId,
        "station_id": req.boardStationId,
        "plate_no": [a.plateNo for a in req.arrivals],
        "pred_sec": [max((naive(a.predictedBusArrivalAt) - at).total_seconds(), 0) for a in req.arrivals],
        "seat_now": [a.remainSeats if a.remainSeats is not None else -1 for a in req.arrivals],
    })
    df = features.add_ahead(df).sort_index()   # 요청 순서로 되돌린다

    day_type = "weekend" if at.weekday() >= 5 else "weekday"
    try:
        if state["seat"].get(day_type):
            seats = model.predict_seats(boosters["seat"], df, capacity["seat"])
        else:   # 예측하지 않은 좌석은 null → 백엔드가 받은 값을 그대로 쓰고 OBSERVED로 표시
            seats = np.full(len(df), np.nan)
        if state["arrival"].get(day_type):
            arrivals = list(model.predict_arrival(boosters["arrival"], df, capacity["arrival"]))
        else:
            arrivals = [naive(a.predictedBusArrivalAt) for a in req.arrivals]
    except Exception as e:
        raise ModelError(f"예측에 실패했어요: {e}")

    predictions = [
        Prediction(plateNo=a.plateNo,
                   predictedBusArrivalAt=kst(pd.Timestamp(arr).floor("s").to_pydatetime()),
                   predictedRemainSeats=None if np.isnan(s) else int(s))
        for a, arr, s in zip(req.arrivals, arrivals, seats)
    ]
    return PredictResponse(predictions=predictions, modelVersion=models.version())
