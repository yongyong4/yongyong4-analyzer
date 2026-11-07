"""매일 학습: 데이터 받기 → 정답 → 하루씩 시험 → 합격한 것만 사용.

실행: .venv/bin/python -m app.train [--skip-download]

합격 판정은 모델 종류(좌석·도착) × 평일·주말 4칸을 따로 한다 (FR-DA-07, FR-AN-02, NFR-05).
  - 기준선 2개(GBIS 값 그대로, 과거 평균)보다 오차가 모두 낮다 (FR-DA-06)
  - 좌석은 '만석인데 여유' 비율도 두 기준선보다 낮다
  - 평일은 시험한 날의 80% 이상에서 GBIS 기준선보다 낫다 (주말은 시험 날짜가 적어 제외)
  - 사용 중 모델보다 3% 넘게 나빠지지 않는다
합격한 칸이 하나라도 있으면 새 버전을 저장하고, 모델 종류별로 사용 중 버전을 바꾼다.
노선별 성적은 기록만 한다.
"""
import argparse
import json
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from app import collector, config, db, features, labels, model

COMMON = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
              min_child_samples=30, verbose=-1, random_state=0)
SEAT_PARAMS = dict(objective="quantile", alpha=0.7)   # 보수적 예측: 감소량을 크게 잡는다
MIN_TRAIN_DAYS = 3
STABLE_RATIO = 0.8
MAX_WORSE = 1.03
KINDS = ("seat", "arrival")
DAY_TYPES = ("weekday", "weekend")
KIND_NAME = {"seat": "좌석", "arrival": "도착"}
DAY_NAME = {"weekday": "평일", "weekend": "주말"}


def fit(df: pd.DataFrame, capacity: dict):
    """좌석 모델은 감소량(현재 - 도착 시점), 도착 모델은 GBIS 오차(분)를 학습한다."""
    x = features.make(df, capacity)
    s = (df.seat_now >= 0) & (df.seat_at >= 0)
    seat = lgb.LGBMRegressor(**COMMON, **SEAT_PARAMS).fit(x[s], (df.seat_now - df.seat_at)[s])
    arrival = lgb.LGBMRegressor(**COMMON).fit(x, df.err_min)
    return seat, arrival


def _groups(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"route_id": df.route_id, "station_id": df.station_id,
                         "slot": df.t.dt.hour * 2 + df.t.dt.minute // 30,
                         "ahead": (df.pred_sec // 300).clip(upper=8),   # 남은 시간 5분 단위 (40분 이상은 한 묶음)
                         "weekend": df.t.dt.day_of_week >= 5}, index=df.index)


def average_baseline(tr: pd.DataFrame, te: pd.DataFrame):
    """과거 평균 기준선: 노선 × 정류장 × 30분 시간대 × 남은 시간(5분) × 평일/주말의 평균 감소량·평균 오차.
    그 묶음이 학습 데이터에 없으면 시간대를 뺀 평균, 그것도 없으면 0."""
    g_tr, g_te = _groups(tr), _groups(te)
    s = (tr.seat_now >= 0) & (tr.seat_at >= 0)
    target = pd.DataFrame({"drop": (tr.seat_now - tr.seat_at).where(s), "err": tr.err_min}, index=tr.index)

    def lookup(col):
        fine_keys = ["route_id", "station_id", "slot", "ahead", "weekend"]
        coarse_keys = ["route_id", "station_id", "ahead", "weekend"]
        fine = target[col].groupby([g_tr[k] for k in fine_keys]).mean()
        coarse = target[col].groupby([g_tr[k] for k in coarse_keys]).mean()
        v = pd.MultiIndex.from_frame(g_te[fine_keys]).map(fine.to_dict().get)
        c = pd.MultiIndex.from_frame(g_te[coarse_keys]).map(coarse.to_dict().get)
        return pd.Series(v, index=te.index, dtype=float).fillna(pd.Series(c, index=te.index, dtype=float)).fillna(0)

    seat_now = te.seat_now.where(te.seat_now >= 0).astype(float)
    seats = np.clip(np.round(seat_now - lookup("drop")), 0, seat_now).to_numpy()
    arrival_at = te.pred_at + pd.to_timedelta(lookup("err"), unit="m")
    return seats, arrival_at


def predictions(tr, te, daily_cap, days_tr) -> pd.DataFrame:
    """시험 날짜 1일에 대해 모델·GBIS·과거 평균 예측을 붙인다."""
    cap = labels.capacity(daily_cap, days_tr)
    seat, arrival = fit(tr, cap)
    m_seats, m_arr = model.apply(seat, arrival, te, cap)
    a_seats, a_arr = average_baseline(tr, te)
    out = te[["day", "route_id", "seat_now", "seat_at", "actual_at", "err_min"]].copy()
    out["weekend"] = te.t.dt.day_of_week >= 5
    out["seat_model"], out["seat_avg"], out["seat_api"] = m_seats, a_seats, te.seat_now.astype(float)
    out["arr_model"] = (te.actual_at - m_arr).dt.total_seconds().abs() / 60
    out["arr_avg"] = (te.actual_at - a_arr).dt.total_seconds().abs() / 60
    out["arr_api"] = te.err_min.abs()
    return out


def metrics(p: pd.DataFrame) -> dict:
    """좌석: MAE·만석인데 여유 비율, 도착: MAE. 각각 모델 / GBIS(api) / 과거 평균(avg)."""
    s = p[(p.seat_now >= 0) & (p.seat_at >= 0)]
    risk = features.seat_stage(s.seat_at) == "RISK"
    m = {"n": int(len(p))}
    for src in ("model", "api", "avg"):
        pred = s[f"seat_{src}"].to_numpy()
        m[f"seat_mae_{src}"] = float(np.abs(pred - s.seat_at.to_numpy()).mean())
        m[f"risk_as_ok_{src}"] = float(((features.seat_stage(pred) == "OK") & risk).sum() / max(risk.sum(), 1))
        m[f"arrival_mae_{src}"] = float(p[f"arr_{src}"].mean())
    return m


def evaluate(df: pd.DataFrame, daily_cap: pd.DataFrame) -> dict:
    """하루씩 늘려 가며 학습하고 다음 날로 시험한다. 정원 추정·과거 평균도 학습 날짜만 쓴다."""
    days = sorted(df.day.unique())
    preds = [predictions(df[df.day.isin(days[:k])], df[df.day == days[k]], daily_cap, days[:k])
             for k in range(MIN_TRAIN_DAYS, len(days))]
    per_day = [dict(day=str(p.day.iloc[0]), weekend=bool(p.weekend.iloc[0]), **metrics(p)) for p in preds]
    allp = pd.concat(preds, ignore_index=True)

    result = {"per_day": per_day}
    for dt in DAY_TYPES:
        rows = [r for r in per_day if r["weekend"] == (dt == "weekend")]
        if not rows:
            continue
        keys = [k for k in rows[0] if k not in ("day", "weekend", "n")]
        result[dt] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
        result[dt]["days"] = len(rows)
        for kind in KINDS:
            result[dt][f"{kind}_stable"] = round(sum(r[f"{kind}_mae_model"] < r[f"{kind}_mae_api"] for r in rows)
                                                 / len(rows), 4)
        # 노선별 성적 (기록용, 판정에 쓰지 않음)
        sub = allp[allp.weekend == (dt == "weekend")]
        result[dt]["by_route"] = {config.ROUTES.get(r, r): {k: round(v, 4) if isinstance(v, float) else v
                                                            for k, v in metrics(g).items()}
                                  for r, g in sub.groupby("route_id")}
    return result


def judge(result: dict, kind: str, dt: str, prev: dict | None) -> list[str]:
    """한 칸(모델 종류 × 평일/주말)이 합격하지 못한 이유. 비어 있으면 합격."""
    m = result.get(dt)
    if not m:
        return ["시험 데이터 없음"]
    reasons = []
    checks = [("mae", "오차")] + ([("risk_as_ok", "만석인데 여유 비율")] if kind == "seat" else [])
    for metric, label in checks:
        key = f"{kind}_{metric}" if metric == "mae" else metric
        for base, base_name in (("api", "GBIS 값"), ("avg", "과거 평균")):
            if not m[f"{key}_model"] < m[f"{key}_{base}"]:
                reasons.append(f"{label}가 {base_name}보다 낮지 않음 ({m[f'{key}_model']:.3f} ≥ {m[f'{key}_{base}']:.3f})")
    if dt == "weekday" and m[f"{kind}_stable"] < STABLE_RATIO:
        reasons.append(f"GBIS 값보다 나은 날 비율 {m[f'{kind}_stable']:.0%} < {STABLE_RATIO:.0%}")
    if prev and prev.get(dt):
        before = prev["metrics"][dt][f"{kind}_mae_model"]
        if m[f"{kind}_mae_model"] > before * MAX_WORSE:
            reasons.append(f"사용 중 모델({prev['version']})보다 3% 넘게 나빠짐 "
                           f"({m[f'{kind}_mae_model']:.3f} > {before:.3f})")
    return reasons


def next_version(today: str) -> str:
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    n = sum(1 for d in os.listdir(config.MODEL_DIR)
            if d.startswith("v") and os.path.isdir(os.path.join(config.MODEL_DIR, d)))
    return f"v{n + 1}-{today}"


def append_history(row: dict) -> None:
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(os.path.join(config.MODEL_DIR, "history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run(skip_download: bool = False) -> dict:
    run_at = datetime.now()
    raw_dir = os.path.join(config.DATA_DIR, "raw")
    if not skip_download:
        if not config.COLLECTOR_BASE_URL:
            raise SystemExit("COLLECTOR_BASE_URL이 없어요 (.env 확인)")
        raw_dir = collector.download()

    df = labels.build(raw_dir)
    daily_cap = labels.daily_capacity(raw_dir)
    result = evaluate(df, daily_cap)

    state = model.active()
    prev = {}
    for kind in KINDS:   # 종류별 사용 중 버전의 성적 + 그때 합격한 칸
        if kind in state:
            meta = model.load(state[kind]["version"])[2]
            prev[kind] = dict(version=meta["version"], metrics=meta["metrics"],
                              **{dt: state[kind].get(dt, False) for dt in DAY_TYPES})
    decisions = {kind: {dt: judge(result, kind, dt, prev.get(kind)) for dt in DAY_TYPES} for kind in KINDS}
    passed = {kind: {dt: not decisions[kind][dt] for dt in DAY_TYPES} for kind in KINDS}

    version = None
    if any(passed[k][d] for k in KINDS for d in DAY_TYPES):
        version = next_version(run_at.strftime("%Y%m%d"))
        cap = labels.capacity(daily_cap)
        seat, arrival = fit(df, cap)
        meta = dict(version=version, trained_at=run_at, data_from=str(min(df.day)), data_to=str(max(df.day)),
                    days=int(df.day.nunique()), samples=int(len(df)), features=features.FEATURES,
                    params=dict(common=COMMON, seat=SEAT_PARAMS), metrics=result,
                    passed=passed, reasons=decisions, capacity=cap)
        model.save(version, seat, arrival, meta)
        # 저장한 파일을 다시 읽어 같은 결과가 나오는지 확인 (서버는 파일로 불러오므로)
        s2, a2, _ = model.load(version)
        sample = df.tail(200)
        assert np.allclose(model.apply(seat, arrival, sample, cap)[0], model.apply(s2, a2, sample, cap)[0],
                           equal_nan=True), "저장한 모델을 다시 읽은 결과가 달라요"
        for kind in KINDS:   # 합격한 칸이 있는 종류만 새 버전으로 바꾼다
            if any(passed[kind].values()):
                state[kind] = {"version": version, **passed[kind]}
        model.set_active(state)

    summary = {dt: {k: v for k, v in result[dt].items() if k != "by_route"} for dt in DAY_TYPES if dt in result}
    append_history(dict(run_at=run_at, version=version, data_to=str(max(df.day)), passed=passed,
                        reasons=decisions, active=state, **summary))

    saved = None
    if db.enabled():   # DB 쓰기가 실패해도 학습 결과(모델 파일)는 그대로 둔다
        try:
            rows = db.bus_pass_rows(df, labels.station_seq(raw_dir), labels.capacity(daily_cap))
            saved = dict(bus_pass=db.write_bus_pass(rows),
                         model_eval=db.write_model_eval(run_at, version, result, passed, state))
        except Exception as e:
            saved = dict(error=str(e))
    return dict(version=version, passed=passed, reasons=decisions, active=state, result=result, db=saved)


def main():
    parser = argparse.ArgumentParser(description="예측 모델 학습")
    parser.add_argument("--skip-download", action="store_true", help="수집 서버에서 받지 않고 data/raw 그대로 사용")
    out = run(parser.parse_args().skip_download)
    r = out["result"]

    print()
    for dt in DAY_TYPES:
        m = r.get(dt)
        if not m:
            continue
        print(f"[{DAY_NAME[dt]} 시험 {m['days']}일]  GBIS 값 / 과거 평균 / 모델")
        print(f"  좌석 오차      {m['seat_mae_api']:.2f} / {m['seat_mae_avg']:.2f} / {m['seat_mae_model']:.2f}석")
        print(f"  만석인데 여유  {m['risk_as_ok_api']:.0%} / {m['risk_as_ok_avg']:.0%} / {m['risk_as_ok_model']:.0%}")
        print(f"  도착 오차      {m['arrival_mae_api']:.2f} / {m['arrival_mae_avg']:.2f} / {m['arrival_mae_model']:.2f}분")
        for route, rm in m["by_route"].items():
            print(f"    {route:<10} 좌석 {rm['seat_mae_api']:.2f} → {rm['seat_mae_model']:.2f}석, "
                  f"도착 {rm['arrival_mae_api']:.2f} → {rm['arrival_mae_model']:.2f}분 ({rm['n']}건)")
    print("\n합격 판정")
    for kind in KINDS:
        for dt in DAY_TYPES:
            ok = out["passed"][kind][dt]
            print(f"  {KIND_NAME[kind]} {DAY_NAME[dt]}: {'✅ 사용' if ok else '❌ GBIS 값 사용'}")
            for reason in out["reasons"][kind][dt]:
                print(f"     - {reason}")
    print(f"\n새 버전: {out['version'] or '없음 (기존 유지)'}")
    print("사용 중:", json.dumps(out["active"], ensure_ascii=False))
    if out["db"] is None:
        print("DB: 저장 안 함 (MYSQL_HOST 없음)")
    elif "error" in out["db"]:
        print(f"DB: 저장 실패 — {out['db']['error']}")
    else:
        print(f"DB: bus_pass {out['db']['bus_pass']}건, model_eval {out['db']['model_eval']}건 저장")


if __name__ == "__main__":
    main()
