# Yongyong Analyzer

용인 대중교통 이동 전략 추천 서비스의 예측·분석 서버입니다.
수집기(yongyong4-collector)가 쌓은 광역·마을버스 데이터로 **도착 시각 보정**과 **도착 시점 잔여좌석 예측** 모델을 학습하고, 백엔드에 `POST /predict`를 제공합니다.

## 개발 환경

- Python 3.12 (uv로 가상환경 생성)
- pandas, scikit-learn, LightGBM
- FastAPI (`/predict`): 사용 중 모델로 좌석·도착 예측 (판정에 떨어진 칸은 받은 값을 그대로 돌려줌)
- MariaDB/MySQL (service DB 분석 테이블 `bus_pass`, `model_eval`): 학습할 때마다 저장. `slot_stat`은 개발 예정

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
   - DB에 저장하려면 서비스 DB에서 테이블을 최초 1회 만듭니다. (백엔드 `sql/trip_schema.sql`과 같은 방식)
     ```bash
     mysql -u root -p yongyong < sql/schema.sql
     ```
4. 수집 서버에서 학습 데이터를 받습니다. (`data/raw/`에 CSV 저장)
   ```bash
   .venv/bin/python -c "from app import collector; collector.download()"
   ```

5. 모델을 학습합니다. (`models/v{번호}-{날짜}/`에 저장, 통과하면 사용 중 버전 교체)
   ```bash
   .venv/bin/python -m app.train                  # 수집 서버에서 받고 학습
   .venv/bin/python -m app.train --skip-download  # 이미 받은 data/raw로 학습
   ```
   - 합격 판정은 모델 종류(좌석·도착) × 평일·주말 4칸을 따로 함. 합격한 칸만 사용하고, 나머지는 GBIS 값 그대로
   - 합격 조건: GBIS 값과 과거 평균, 두 기준선보다 오차가 낮음 (좌석은 '만석인데 여유' 비율도). 평일은 시험한 날의 80% 이상에서 GBIS 값보다 나음. 사용 중 모델보다 3% 넘게 나빠지지 않음
   - 노선별 성적은 기록만 함 (`meta.json`)
   - 실행할 때마다 결과를 `models/history.jsonl`에 한 줄씩 기록
   - `MYSQL_HOST`가 있으면 DB에도 저장: `bus_pass`(실제 통과 기록), `model_eval`(좌석·도착 × 평일·주말 성적, 사용 중 여부). 다시 실행해도 `bus_pass`는 중복되지 않음
   - 수집 시간대(평일 06:00-10:30, 주말 10:00-15:00)를 피해서 실행
6. 예측 서버를 실행합니다.
   ```bash
   .venv/bin/uvicorn app.server:app --port 8000
   ```
   - http://localhost:8000/health → `status: UP`, 사용 중 모델 버전
   - 매일 학습으로 `models/active.json`이 바뀌면 재시작 없이 다음 요청부터 새 모델 사용
   - 모델이 없거나 예측에 실패하면 500 → 백엔드는 받은 값을 그대로 씀 (OBSERVED)
   - http://localhost:8000/docs → `/predict` 요청을 화면에서 직접 보내 볼 수 있음
   - 백엔드는 `PREDICT_BASE_URL=http://localhost:8000`으로 연결

## 환경변수 (`.env`)

| 이름 | 설명 |
|---|---|
| `COLLECTOR_BASE_URL` | 수집 서버 주소. 팀 채널에서 공유 |
| `DATA_FROM` | 학습 데이터 시작일 (수집 시작 2026-10-28) |
| `DATA_DIR` | 내려받은 CSV·정답 저장 위치 (기본 `data`) |
| `MODEL_DIR` | 학습한 모델 저장 위치 (기본 `models`) |
| `MYSQL_HOST`, `MYSQL_PORT` | 서비스 DB 주소 (비우면 DB에 저장 안 함) |
| `MYSQL_DATABASE` | DB 이름 (기본 `yongyong`) |
| `MYSQL_USER`, `MYSQL_PASSWORD` | DB 계정 (백엔드와 같은 값) |

## 서비스 DB (분석 테이블)

- 테이블 생성 SQL: `sql/schema.sql` (`bus_pass`, `model_eval`). 서비스 DB에서 **최초 1회** 실행합니다.
  ```bash
  mysql -u root -p yongyong < sql/schema.sql
  ```
- 백엔드 테이블(`Yongyong4-backend/sql/trip_schema.sql`)과 같은 방식(`CREATE TABLE IF NOT EXISTS`)이라 다시 실행해도 안전합니다.
- 백엔드는 이 테이블을 읽지 않고, `/predict`도 DB를 쓰지 않습니다. 테이블이 없어도 서비스는 정상 동작하고, 학습 결과만 DB에 남지 않습니다.

| 테이블 | 언제 | 내용 |
|---|---|---|
| `bus_pass` | 학습할 때마다 (전체 기간 다시 저장, 중복 없음) | 버스 1대가 정류장 1곳을 실제로 지난 시각·좌석 (AI 정답) |
| `model_eval` | 학습할 때마다 4줄 추가 | 좌석·도착 × 평일·주말 성적(기준선 = GBIS 값), 통과 여부, 사용 중 여부 (`models/active.json`과 같음) |

**ERD와 다른 점**
- `model_eval.day_type` 추가 (`WEEKDAY` / `WEEKEND`, `slot_stat.day_type`과 같은 값): 평일·주말을 따로 판정하기 때문
- 인덱스 `idx_model_active`에 `day_type` 추가
- 새 버전을 만들지 않은 날(모두 불합격)은 `model_version = '미채택'`, `model_path = NULL`로 기록

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
  train.py       학습 → 하루씩 시험 → 좌석·도착 × 평일·주말 합격 판정 → 새 버전 저장
  server.py      FastAPI POST /predict
  db.py          서비스 DB에 bus_pass, model_eval 저장
sql/schema.sql   분석 테이블 생성 (최초 1회)
experiments/     데이터 분석·모델 비교 스크립트
data/            내려받은 CSV·정답 (git 제외)
models/          학습한 모델 (git 제외)
```

## 이후 작업

1. `stats.py`: 광역·마을버스 배차간격 `slot_stat`
2. 배포 서버에 매일 재학습 cron 등록
