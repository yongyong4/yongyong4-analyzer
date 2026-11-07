-- 용용톤 서비스 DB: 분석 테이블 (파이썬 예측 서버가 씀)
-- MySQL / MariaDB 기준, 서비스 DB에서 최초 1회 실행
-- ERD 기준. model_eval에만 day_type 추가 (좌석·도착 × 평일·주말을 따로 판정하기 때문)

CREATE TABLE IF NOT EXISTS bus_pass (
    pass_id           BIGINT       NOT NULL AUTO_INCREMENT COMMENT '통과 ID',
    service_date      DATE         NOT NULL COMMENT '운행일',
    route_id          VARCHAR(20)  NOT NULL COMMENT '노선 ID',
    plate_no          VARCHAR(20)  NOT NULL COMMENT '차량번호',
    run_seq           INT          NOT NULL COMMENT '운행 회차',
    station_seq       INT          NOT NULL COMMENT '정류장 순번',
    station_id        VARCHAR(20)  NOT NULL COMMENT '정류장 ID',
    pass_at           DATETIME     NOT NULL COMMENT '통과 시각',
    remain_seat       INT          NULL     COMMENT '통과 시 남은 좌석 (AI 정답)',
    is_double_decker  BOOLEAN      NULL     COMMENT '2층버스 여부',
    PRIMARY KEY (pass_id),
    UNIQUE KEY uk_bus_pass (service_date, route_id, plate_no, run_seq, station_seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='통과 기록: 버스 1대가 정류장 1곳을 실제로 지난 기록 (파이썬, 하루 1번)';

CREATE TABLE IF NOT EXISTS model_eval (
    eval_id        BIGINT        NOT NULL AUTO_INCREMENT COMMENT '평가 ID',
    model_type     VARCHAR(20)   NOT NULL COMMENT '모델 종류: ARRIVAL 도착시간 보정 / SEAT 잔여좌석 예측',
    day_type       VARCHAR(20)   NOT NULL COMMENT '요일 구분: WEEKDAY 평일 / WEEKEND 주말',
    model_version  VARCHAR(50)   NOT NULL COMMENT '모델 버전',
    route_id       VARCHAR(20)   NULL     COMMENT '노선 ID (NULL = 전체)',
    baseline_mae   DECIMAL(8,3)  NULL     COMMENT '기준선 MAE (GBIS 값 그대로)',
    model_mae      DECIMAL(8,3)  NULL     COMMENT '모델 MAE',
    is_passed      BOOLEAN       NOT NULL COMMENT '검증 통과',
    is_active      BOOLEAN       NOT NULL DEFAULT FALSE COMMENT '사용 중',
    model_path     VARCHAR(255)  NULL     COMMENT '모델 파일 위치',
    evaluated_at   DATETIME      NOT NULL COMMENT '평가 시각',
    PRIMARY KEY (eval_id),
    INDEX idx_model_active (model_type, day_type, route_id, is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='모델 성적표: 모델 1개 버전의 시험 결과 (파이썬, 학습할 때마다)';
