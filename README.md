# Yongyong Analyzer

용인 대중교통 이동 전략 추천 서비스의 예측·분석 서버입니다.
수집기(yongyong4-collector)가 쌓은 광역·마을버스 데이터로 **도착 시각 보정**과 **도착 시점 잔여좌석 예측** 모델을 학습하고, 백엔드에 `POST /predict`를 제공합니다.

## 개발 환경

- Python 3.12 (uv로 가상환경 생성)
- pandas, scikit-learn, LightGBM
- FastAPI (`/predict`): **개발 예정**
- MariaDB (service DB 분석 테이블 `bus_pass`, `slot_stat`, `model_eval`): **개발 예정**

## 초기 실행

1. uv가 설치돼 있는지 확인합니다: `uv --version` (없으면 `brew install uv`)
2. 가상환경을 만들고 라이브러리를 설치합니다.
   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -r requirements.txt
   ```
3. 환경변수 파일을 만들고 값을 채웁니다.
   ```bash
   cp .env.example .env
   ```
4. 수집 서버에서 학습 데이터를 받습니다. (`data/raw/`에 CSV 저장)
   ```bash
   .venv/bin/python -c "from app import collector; collector.download()"
   ```

## 환경변수 (`.env`)

| 이름 | 설명 |
|---|---|
| `COLLECTOR_BASE_URL` | 수집 서버 주소. 팀 채널에서 공유 |
| `DATA_FROM` | 학습 데이터 시작일 (수집 시작 2026-10-28) |
| `DATA_DIR` | 내려받은 CSV·정답 저장 위치 (기본 `data`) |
| `MODEL_DIR` | 학습한 모델 저장 위치 (기본 `models`) |

## 예측 대상

| 노선 | routeId |
|---|---|
| 5000 | 228000174 |
| 5001A | 228000429 |
| 5003A | 228000431 |
| 5005 | 228000175 |
| 5005(거점) | 228000447 |

정류장: 시청.용인대역(228000745), 동백역(228000172), 강남대역(228000684), 기흥역(228000682), 상미마을(228000680)

- 수집 시간대(평일 06:00-10:30, 주말 10:00-15:00)만 예측합니다. 대상 밖 노선·정류장·시간대는 400을 돌려주고, 백엔드는 GBIS 값을 그대로 씁니다(`seatType = OBSERVED`).
- 마을버스는 도착정보만 수집돼서 좌석 예측은 하지 않고, 배차간격 통계(`slot_stat`)에만 씁니다.

## 실험 (현재 결과)

```bash
.venv/bin/python experiments/compare_models.py data/labels/labels.csv     # 모델 비교
.venv/bin/python experiments/village_pass_check.py data/raw data/labels/labels_v2.csv   # 마을버스 통과 추정 검증
```

2026-10-28부터 11-06까지 평일 8일, 하루씩 늘려 가며 다음 날로 시험한 결과입니다.

| 항목 | 기준선 (GBIS 값 그대로) | LightGBM |
|---|---|---|
| 잔여좌석 오차 (MAE) | 12.9석 | 3.5석 |
| 만석인데 "여유"로 안내한 비율 | 61% | 12% (보수적 예측) |
| 도착 시각 오차 (MAE) | 2.35분 | 1.75분 |

## 폴더 구조

```text
app/
  config.py      대상 노선·정류장, 수집 시간대, 환경변수
  collector.py   수집 서버 /export/zip 에서 CSV 받기
  labels.py      같은 차량을 따라가 실제 통과 시각·도착 시점 좌석(정답) 만들기
  features.py    특징값 (학습과 /predict가 같은 함수 사용)
  model.py       모델 적용·저장·불러오기
experiments/     데이터 분석·모델 비교 스크립트
data/            내려받은 CSV·정답 (git 제외)
models/          학습한 모델 (git 제외)
```

## 이후 작업

1. `train.py`: 매일 학습 → 평가 → 기준선·현재 모델보다 나을 때만 새 버전 저장
2. `server.py`: FastAPI `POST /predict` (명세 「도착·좌석 예측」)
3. `db.py`: service DB에 `bus_pass`, `model_eval` 쓰기
4. `stats.py`: 광역·마을버스 배차간격 `slot_stat`
5. 배포 서버에 매일 재학습 cron 등록
