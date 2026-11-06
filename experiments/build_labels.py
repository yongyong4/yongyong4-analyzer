"""정답값(실제 도착 시각·도착 시점 좌석) 생성 가능성 + GBIS 원래 값 오차 확인."""
import sys, glob, os
import numpy as np
import pandas as pd

RAW = sys.argv[1]
ROUTES = {"228000174": "5000", "228000429": "5001A", "228000431": "5003A",
          "228000175": "5005", "228000447": "5005(거점)"}
STATIONS = {"228000745": "시청.용인대역", "228000172": "동백역", "228000684": "강남대역",
            "228000682": "기흥역", "228000680": "상미마을"}

rs = pd.read_csv(os.path.join(RAW, "route_station.csv"), dtype=str)
rs["station_seq"] = rs["station_seq"].astype(int)
rs = rs[rs.route_id.isin(ROUTES) & rs.station_id.isin(STATIONS)]
seq = rs.groupby(["route_id", "station_id"]).station_seq.min().to_dict()

loc = pd.concat([pd.read_csv(f, dtype={"route_id": str, "station_id": str})
                 for f in sorted(glob.glob(os.path.join(RAW, "bus_location_*.csv")))])
arr = pd.concat([pd.read_csv(f, dtype={"route_id": str, "station_id": str})
                 for f in sorted(glob.glob(os.path.join(RAW, "bus_arrival_*.csv")))])
for d in (loc, arr):
    d["t"] = pd.to_datetime(d.collected_at).dt.floor("s")
    d["day"] = d.t.dt.date
arr = arr[arr.route_id.isin(ROUTES) & arr.station_id.isin(STATIONS) & arr.plate_no.notna()]
arr = arr[[(r, s) in seq for r, s in zip(arr.route_id, arr.station_id)]]
print(f"대상 노선 도착예정 행: {len(arr):,}  위치 행: {len(loc):,}")
print("노선×정류장 조합:", len(seq))

loc = loc.sort_values("t")
groups = {k: g for k, g in loc.groupby(["day", "route_id", "plate_no"])}

rows = []
for r in arr.itertuples():
    g = groups.get((r.day, r.route_id, r.plate_no))
    if g is None:
        continue
    s = seq[(r.route_id, r.station_id)]
    g = g[(g.t >= r.t - pd.Timedelta(minutes=3)) & (g.t <= r.t + pd.Timedelta(minutes=90))]
    after = g[(g.t >= r.t) & (g.station_seq >= s)]
    if after.empty:
        continue
    first = after.iloc[0]
    if first.station_seq > s + 5:          # 너무 멀리 건너뜀 = 다른 회차 의심
        continue
    before = g[(g.t < first.t) & (g.station_seq < s)]
    if first.station_seq == s or before.empty:
        actual = first.t
    else:
        actual = before.iloc[-1].t + (first.t - before.iloc[-1].t) / 2
    if before.empty:
        continue
    seat_at = before.iloc[-1].remain_seat_cnt   # 정류장 직전 마지막 관측 좌석
    rows.append(dict(day=r.day, t=r.t, route=ROUTES[r.route_id], station=STATIONS[r.station_id],
                     order=r.arrival_order, pred_sec=r.predict_time_sec, pred_min=r.predict_time_min,
                     seat_now=r.remain_seat_cnt, actual=actual, seat_at=seat_at,
                     gap_sec=(first.t - before.iloc[-1].t).total_seconds()))

df = pd.DataFrame(rows)
df["pred_at"] = df.t + pd.to_timedelta(df.pred_sec, unit="s")
df["err_min"] = (df.actual - df.pred_at).dt.total_seconds() / 60
df["weekend"] = pd.to_datetime(df.day).dt.dayofweek >= 5
df["ahead"] = pd.cut(df.pred_min, [-1, 5, 10, 20, 40, 200], labels=["~5", "6~10", "11~20", "21~40", "40~"])
print(f"\n정답 붙은 샘플: {len(df):,} / {len(arr):,} ({len(df)/len(arr):.0%})")
print("정답 시각 해상도(직전~직후 관측 간격) 중앙값:", df.gap_sec.median(), "초")
print(df.groupby("weekend").size().rename("샘플수"))

wd = df[~df.weekend]
print("\n=== [평일] 도착 시각: GBIS 예측 오차 (실제 - 예측, 분) ===")
print(f"MAE {wd.err_min.abs().mean():.2f}분 / 평균 {wd.err_min.mean():+.2f}분 (+ = 실제가 늦음)")
print(wd.groupby("ahead", observed=True).err_min.agg(샘플="size", MAE=lambda x: x.abs().mean(), 평균편향="mean").round(2))
print(wd.groupby("station").err_min.agg(샘플="size", MAE=lambda x: x.abs().mean(), 평균편향="mean").round(2))

st = wd[(wd.seat_now >= 0) & (wd.seat_at >= 0)].copy()
st["drop"] = st.seat_now - st.seat_at
stage = lambda x: np.where(x >= 5, "OK", np.where(x >= 1, "CAUTION", "RISK"))
st["stage_now"], st["stage_at"] = stage(st.seat_now), stage(st.seat_at)
print(f"\n=== [평일] 좌석: 현재 좌석을 그대로 쓸 때 (샘플 {len(st):,}) ===")
print(f"MAE {(st.seat_now - st.seat_at).abs().mean():.2f}석 / 평균 감소 {st['drop'].mean():.2f}석")
print(st.groupby("ahead", observed=True)["drop"].agg(샘플="size", 평균감소="mean", MAE=lambda x: x.abs().mean()).round(2))
print(st.groupby("station")["drop"].agg(샘플="size", 평균감소="mean", MAE=lambda x: x.abs().mean()).round(2))
print("\n단계 정확도:", f"{(st.stage_now == st.stage_at).mean():.1%}")
print("실제 RISK(0석) 샘플:", (st.stage_at == "RISK").sum(),
      "/ 그중 현재값으로 OK라고 한 비율:",
      f"{((st.stage_at == 'RISK') & (st.stage_now == 'OK')).sum() / max((st.stage_at == 'RISK').sum(), 1):.1%}")
print(pd.crosstab(st.stage_at, st.stage_now, rownames=["실제"], colnames=["현재값"]))
print("\n일별 평일 샘플:", wd.groupby("day").size().to_dict())
df.to_csv(sys.argv[2], index=False)
