"""POST /predict — 명세 「도착·좌석 예측」.

지금은 stub: 요청 검증(400 규칙)만 하고 받은 값을 그대로 돌려준다.
train.py가 완성되면 '그대로 돌려주는' 부분만 모델 예측으로 바꾼다.
실행: .venv/bin/uvicorn app.server:app --port 8000
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import config

KST = ZoneInfo("Asia/Seoul")
MODEL_VERSION = "stub"

app = FastAPI(title="Yongyong Analyzer")


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


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, e: RequestValidationError):
    # 명세: 필수값 누락은 400 (FastAPI 기본은 422)
    return JSONResponse(status_code=400, content={"message": "필수값이 없거나 형식이 맞지 않아요", "detail": str(e.errors())})


@app.exception_handler(BadRequest)
async def bad_request(_: Request, e: BadRequest):
    return JSONResponse(status_code=400, content={"message": str(e)})


def kst(dt: datetime) -> datetime:
    return dt.replace(tzinfo=KST) if dt.tzinfo is None else dt.astimezone(KST)


@app.get("/health")
def health():
    return {"status": "UP", "modelVersion": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.gbisRouteId not in config.ROUTES:
        raise BadRequest("수집 광역 노선이 아니에요")
    if req.boardStationId not in config.STATIONS:
        raise BadRequest("수집 정류장이 아니에요")
    if not config.in_window(kst(req.requestAt).replace(tzinfo=None)):
        raise BadRequest("수집 시간대가 아니에요")

    # stub: 받은 값을 그대로 돌려준다
    predictions = [
        Prediction(plateNo=a.plateNo,
                   predictedBusArrivalAt=kst(a.predictedBusArrivalAt),
                   predictedRemainSeats=a.remainSeats)
        for a in req.arrivals
    ]
    return PredictResponse(predictions=predictions, modelVersion=MODEL_VERSION)
