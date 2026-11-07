"""서비스 DB에 분석 결과 쓰기: bus_pass(통과 기록), model_eval(모델 성적표). 테이블은 sql/schema.sql.

MYSQL_HOST가 없으면 쓰지 않는다 (로컬 학습만 할 때). 같은 데이터를 다시 써도 중복되지 않는다.
"""
import pandas as pd
import pymysql

from app import config

DOUBLE_DECKER_SEATS = 60   # 추정 정원이 이 이상이면 2층버스 (일반 41~50석, 2층 61~80석)
RUN_GAP = pd.Timedelta(minutes=90)
MODEL_TYPE = {"seat": "SEAT", "arrival": "ARRIVAL"}
DAY_TYPE = {"weekday": "WEEKDAY", "weekend": "WEEKEND"}


def enabled() -> bool:
    return bool(config.MYSQL_HOST)


def connect():
    return pymysql.connect(host=config.MYSQL_HOST, port=config.MYSQL_PORT, user=config.MYSQL_USER,
                           password=config.MYSQL_PASSWORD, database=config.MYSQL_DATABASE,
                           charset="utf8mb4", autocommit=False)


def bus_pass_rows(df: pd.DataFrame, seqs: dict, capacity: dict) -> pd.DataFrame:
    """정답 데이터(도착예정 1건 = 1행)를 통과 1건 = 1행으로 줄이고 운행 회차를 붙인다.
    같은 날·노선·차량에서 정류장 순번이 줄거나 90분 넘게 비면 다음 회차."""
    p = df.drop_duplicates(["day", "route_id", "plate_no", "station_id", "actual_at"]).copy()
    p["station_seq"] = [seqs[(r, s)] for r, s in zip(p.route_id, p.station_id)]
    p = p.sort_values(["day", "route_id", "plate_no", "actual_at"])
    g = p.groupby(["day", "route_id", "plate_no"])
    new_run = (g.station_seq.diff() <= 0) | (g.actual_at.diff() > RUN_GAP)
    p["run_seq"] = new_run.astype(int).groupby([p.day, p.route_id, p.plate_no]).cumsum() + 1
    cap = p.plate_no.map(capacity)
    p["is_double_decker"] = (cap >= DOUBLE_DECKER_SEATS).where(cap.notna())
    p["remain_seat"] = p.seat_at.where(p.seat_at >= 0)
    return p.drop_duplicates(["day", "route_id", "plate_no", "run_seq", "station_seq"])


def write_bus_pass(rows: pd.DataFrame) -> int:
    sql = """INSERT INTO bus_pass (service_date, route_id, plate_no, run_seq, station_seq, station_id,
                                   pass_at, remain_seat, is_double_decker)
             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
             ON DUPLICATE KEY UPDATE station_id = VALUES(station_id), pass_at = VALUES(pass_at),
                                     remain_seat = VALUES(remain_seat), is_double_decker = VALUES(is_double_decker)"""
    values = [(r.day, r.route_id, r.plate_no, int(r.run_seq), int(r.station_seq), r.station_id,
               r.actual_at.floor("s").to_pydatetime(),
               None if pd.isna(r.remain_seat) else int(r.remain_seat),
               None if pd.isna(r.is_double_decker) else bool(r.is_double_decker))
              for r in rows.itertuples()]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, values)
        conn.commit()
    return len(values)


def write_model_eval(run_at, version: str | None, result: dict, passed: dict, state: dict) -> int:
    """종류 × 평일·주말 1행씩 (route_id NULL = 전체). 기준선은 GBIS 값 그대로.
    그다음 active.json과 같도록 is_active를 다시 맞춘다."""
    values = []
    for kind, model_type in MODEL_TYPE.items():
        for dt, day_type in DAY_TYPE.items():
            if dt not in result:
                continue
            m = result[dt]
            values.append((model_type, day_type, version or "미채택", None,
                           round(m[f"{kind}_mae_api"], 3), round(m[f"{kind}_mae_model"], 3),
                           passed[kind][dt], False,
                           f"{config.MODEL_DIR}/{version}" if version else None, run_at))
    with connect() as conn, conn.cursor() as cur:
        cur.executemany("""INSERT INTO model_eval (model_type, day_type, model_version, route_id, baseline_mae,
                                                   model_mae, is_passed, is_active, model_path, evaluated_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", values)
        for kind, model_type in MODEL_TYPE.items():
            for dt, day_type in DAY_TYPE.items():
                active = state.get(kind, {})
                cur.execute("""UPDATE model_eval
                               SET is_active = (model_version = %s AND route_id IS NULL AND %s)
                               WHERE model_type = %s AND day_type = %s""",
                            (active.get("version", ""), bool(active.get(dt)), model_type, day_type))
        conn.commit()
    return len(values)
