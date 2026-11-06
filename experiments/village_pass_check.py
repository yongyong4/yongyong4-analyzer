"""도착정보만으로 통과 시각을 추정할 수 있는지 확인 (마을버스용).

방법: 정류장·노선·차량별로 도착예정에 계속 잡히다가 사라지면, 마지막으로 본 시각 + 그때 남은 초 = 통과 추정.
검증: 같은 방법을 광역버스에 적용해, 위치 데이터로 만든 실제 통과 시각(labels.csv)과 비교한다.
사용: .venv/bin/python experiments/village_pass_check.py data/raw data/labels/labels.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

RAW, LABELS = sys.argv[1], sys.argv[2]
EPISODE_GAP = pd.Timedelta(minutes=10)     # 같은 차량이 이보다 오래 안 보이면 다른 회차
IMMINENT_SEC = 180                         # 사라지기 직전 남은 시간이 이 안이면 '통과'로 인정

arr = pd.concat([pd.read_csv(f, dtype={"route_id": str, "station_id": str})
                 for f in sorted(glob.glob(os.path.join(RAW, "bus_arrival_*.csv")))], ignore_index=True)
arr = arr[arr.plate_no.notna()].copy()
arr["t"] = pd.to_datetime(arr.collected_at).dt.floor("s")
arr["day"] = arr.t.dt.date

# 정류장별 수집 주기 시각 (다음 주기 = 사라진 걸 확인한 시각)
cycles = arr[["station_id", "t"]].drop_duplicates().sort_values("t")


def passes(a: pd.DataFrame) -> pd.DataFrame:
    a = a.sort_values(["station_id", "route_id", "plate_no", "t"])
    key = ["day", "station_id", "route_id", "plate_no"]
    new_ep = (a.groupby(key).t.diff() > EPISODE_GAP).fillna(True)
    a["ep"] = new_ep.groupby([a[k] for k in key]).cumsum()
    last = a.groupby(key + ["ep"]).agg(last_t=("t", "max"), n=("t", "size"),
                                       route_name=("route_name", "first"),
                                       route_type=("route_type_cd", "first"))
    last_row = a.loc[a.groupby(key + ["ep"]).t.idxmax(), key + ["ep", "predict_time_sec"]]
    p = last.reset_index().merge(last_row, on=key + ["ep"])
    # 사라진 걸 확인한 다음 주기 시각
    p = pd.merge_asof(p.sort_values("last_t"), cycles.rename(columns={"t": "next_t"}),
                      left_on="last_t", right_on="next_t", by="station_id",
                      direction="forward", allow_exact_matches=False)
    p["pass_est"] = p.last_t + pd.to_timedelta(p.predict_time_sec.fillna(0), unit="s")
    p["pass_est"] = p[["pass_est", "next_t"]].min(axis=1)
    p["imminent"] = p.predict_time_sec <= IMMINENT_SEC
    return p


p = passes(arr)

# --- 1) 광역버스로 방법 검증 ---
lab = pd.read_csv(LABELS, dtype={"route_id": str, "station_id": str})
lab["actual_at"] = pd.to_datetime(lab.actual_at, format="mixed")
truth = lab[["day", "route_id", "station_id", "plate_no", "actual_at"]].drop_duplicates()
truth["day"] = pd.to_datetime(truth.day).dt.date
wide = p[p.route_id.isin(lab.route_id.unique()) & p.station_id.isin(lab.station_id.unique())]
m = wide.merge(truth, on=["day", "route_id", "station_id", "plate_no"])
m["err_min"] = (m.pass_est - m.actual_at).dt.total_seconds() / 60
m = m.loc[m.groupby(["day", "route_id", "station_id", "plate_no", "ep"]).err_min.apply(lambda s: s.abs().idxmin())]
print("=== 1) 광역버스로 검증: 도착정보 추정 vs 위치 데이터 실제 통과 ===")
print(f"추정 통과 {len(wide):,}건 중 실제와 짝지어진 것 {len(m):,}건, 직전 남은시간 3분 이내 비율 {wide.imminent.mean():.1%}")
mi = m[m.imminent]
print(f"[3분 이내에서 사라진 것] 오차 MAE {mi.err_min.abs().mean():.2f}분, 중앙값 {mi.err_min.median():+.2f}분, "
      f"2분 이내 {(mi.err_min.abs() <= 2).mean():.1%}, 4분 이내 {(mi.err_min.abs() <= 4).mean():.1%}")
print(f"[3분 넘게 남기고 사라진 것] {(~m.imminent).sum()}건, MAE {m[~m.imminent].err_min.abs().mean():.2f}분")

# --- 2) 마을버스 적용 ---
v = p[p.route_type == 30].copy()
print(f"\n=== 2) 마을버스: 추정 통과 {len(v):,}건, 3분 이내에서 사라진 비율 {v.imminent.mean():.1%} ===")
v = v[v.imminent].sort_values("pass_est")
v["weekend"] = pd.to_datetime(v.day).dt.dayofweek >= 5
v["gap_min"] = v.groupby(["day", "station_id", "route_id"]).pass_est.diff().dt.total_seconds() / 60
g = v[~v.weekend].groupby(["route_name", "station_id"]).agg(
    통과수=("pass_est", "size"), 일수=("day", "nunique"),
    간격중앙=("gap_min", "median"), 간격최소=("gap_min", "min"), 간격최대=("gap_min", "max"))
g["하루평균통과"] = (g.통과수 / g.일수).round(1)
print("[평일] 노선·정류장별 배차간격(분)")
print(g.round(1).sort_values("통과수", ascending=False).to_string())

# 같은 차량이 짧은 시간에 두 번 '통과'로 잡히는지 (끊겼다 다시 잡힘 = 오탐)
dup = v.groupby(["day", "station_id", "route_id", "plate_no"]).pass_est.diff().dt.total_seconds() / 60
print(f"\n같은 차량이 20분 안에 다시 통과로 잡힌 건수(오탐 의심): {(dup < 20).sum()} / {len(v)}")
