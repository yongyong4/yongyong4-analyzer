"""정답 만들기: 도착예정 1건(그 시점 GBIS 값)에 같은 차량의 실제 통과 시각·좌석을 붙인다.

- 실제 통과 시각: 그 정류장 순번 이상에 처음 관측된 시각.
  정류장에 '도착' 상태로 잡혔으면 그 시각, 지나쳐서 잡혔으면 직전 관측과의 중간.
- 도착 시점 좌석: 정류장 직전 마지막 관측 좌석.
"""
import glob
import os

import pandas as pd

from app import config, features

MAX_TRAVEL = pd.Timedelta(minutes=90)   # 도착예정 후 이 안에 통과해야 같은 회차로 봄
MAX_SKIP_SEQ = 5                        # 처음 관측 순번이 정류장보다 이만큼 넘으면 버림


def _read(raw_dir: str, table: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, f"{table}_*.csv")))
    df = pd.concat([pd.read_csv(f, dtype={"route_id": str, "station_id": str}) for f in files],
                   ignore_index=True)
    df["t"] = pd.to_datetime(df.collected_at).dt.floor("s")
    df["day"] = df.t.dt.date
    return df


def station_seq(raw_dir: str) -> dict:
    rs = pd.read_csv(os.path.join(raw_dir, "route_station.csv"), dtype=str)
    rs = rs[rs.route_id.isin(config.ROUTES) & rs.station_id.isin(config.STATIONS)]
    return rs.assign(station_seq=rs.station_seq.astype(int)) \
             .groupby(["route_id", "station_id"]).station_seq.min().to_dict()


def daily_capacity(raw_dir: str) -> pd.DataFrame:
    """날짜·차량별 관측된 최대 좌석. 정원 추정(2층버스 등 구분)에 쓴다."""
    loc = _read(raw_dir, "bus_location")
    return loc[loc.remain_seat_cnt >= 0].groupby(["day", "plate_no"]).remain_seat_cnt.max().reset_index()


def capacity(daily: pd.DataFrame, days=None) -> dict:
    """주어진 날짜들(None이면 전체)에서 차량별 최대 좌석 = 정원 추정."""
    d = daily if days is None else daily[daily.day.isin(days)]
    return {k: int(v) for k, v in d.groupby("plate_no").remain_seat_cnt.max().items()}


def build(raw_dir: str) -> pd.DataFrame:
    seqs = station_seq(raw_dir)
    loc = _read(raw_dir, "bus_location")
    loc = loc[loc.route_id.isin(config.ROUTES) & loc.plate_no.notna()]
    arr = _read(raw_dir, "bus_arrival")
    arr = arr[arr.route_id.isin(config.ROUTES) & arr.station_id.isin(config.STATIONS)
              & arr.plate_no.notna()]
    # 앞차 정보는 정답 유무와 상관없이 그 시각 도착정보 전체에서 계산한다 (요청의 arrivals와 같게)
    arr = features.add_ahead(arr.rename(columns={"predict_time_sec": "pred_sec", "remain_seat_cnt": "seat_now"}))

    keys = ["day", "route_id", "plate_no"]
    loc = loc[keys + ["t", "station_seq", "remain_seat_cnt"]].sort_values("t")
    out = []
    for (route_id, station_id), s in seqs.items():
        a = arr[(arr.route_id == route_id) & (arr.station_id == station_id)].sort_values("t")
        if a.empty:
            continue
        # 정류장 순번 이상에 처음 관측된 시점
        after = loc[loc.station_seq >= s].rename(columns={"t": "first_t", "station_seq": "first_seq"})
        a = pd.merge_asof(a, after[keys + ["first_t", "first_seq"]], left_on="t", right_on="first_t",
                          by=keys, direction="forward", tolerance=MAX_TRAVEL)
        a = a[a.first_t.notna() & (a.first_seq <= s + MAX_SKIP_SEQ)].sort_values("first_t")
        # 그 직전, 정류장 앞에서 마지막 관측
        before = loc[loc.station_seq < s].rename(columns={"t": "prev_t", "remain_seat_cnt": "seat_at"})
        a = pd.merge_asof(a, before[keys + ["prev_t", "seat_at"]], left_on="first_t", right_on="prev_t",
                          by=keys, direction="backward", allow_exact_matches=False)
        a = a[a.prev_t.notna() & (a.prev_t >= a.t - pd.Timedelta(minutes=3))]
        exact = a.first_seq == s
        a["actual_at"] = a.first_t.where(exact, a.prev_t + (a.first_t - a.prev_t) / 2)
        out.append(a)

    df = pd.concat(out, ignore_index=True)
    df["pred_at"] = df.t + pd.to_timedelta(df.pred_sec, unit="s")
    df["err_min"] = (df.actual_at - df.pred_at).dt.total_seconds() / 60
    return df[
        ["day", "t", "route_id", "station_id", "plate_no", "arrival_order", "pred_sec",
         "seat_now", "ahead_sec", "ahead_seat", "pred_at", "actual_at", "err_min", "seat_at"]
    ].sort_values("t", ignore_index=True)
