"""좌석·도착 예측 방법 비교 (평일, 하루씩 늘려가며 다음 날로 시험).

사용: .venv/bin/python experiments/compare_models.py data/labels/labels.csv
특징값은 /predict 요청에 들어오는 값만 쓴다: 노선, 정류장, 현재 좌석, 남은 초, 시각, 요일.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

CAT = ["route", "station"]
NUM = ["seat_now", "pred_sec", "tod_min", "dow"]

df = pd.read_csv(sys.argv[1], parse_dates=["t", "actual", "pred_at"])
df = df[~df.weekend & (df.seat_now >= 0) & (df.seat_at >= 0)].copy()
df["tod_min"] = df.t.dt.hour * 60 + df.t.dt.minute
df["dow"] = df.t.dt.dayofweek
df["drop"] = df.seat_now - df.seat_at
for c in CAT:
    df[c] = df[c].astype("category")
days = sorted(df.day.unique())


def stage(x):
    return np.where(x >= 5, "OK", np.where(x >= 1, "CAUTION", "RISK"))


def lin():
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT),
                             ("n", StandardScaler(), NUM)])
    return make_pipeline(pre, Ridge(alpha=1.0))


def rf():
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT),
                             ("n", "passthrough", NUM)])
    return make_pipeline(pre, RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=0))


def gbm(**kw):
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=30, verbose=-1, random_state=0, **kw)


def rule_fit(tr, target):
    """노선×정류장×남은시간(5분)×시각(30분) 평균. 그룹이 없으면 노선×정류장 평균."""
    def key(d):
        return (d.route.astype(str) + "|" + d.station.astype(str) + "|"
                + (d.pred_sec // 300).astype(int).astype(str) + "|" + (d.tod_min // 30).astype(str))
    fine = tr.groupby(key(tr))[target].mean()
    coarse = tr.groupby(tr.route.astype(str) + "|" + tr.station.astype(str))[target].mean()

    def predict(te):
        p = key(te).map(fine)
        return p.fillna((te.route.astype(str) + "|" + te.station.astype(str)).map(coarse)).fillna(0).values
    return predict


SEAT_MODELS = {
    "기준선(현재좌석)": None,
    "규칙(그룹평균)": "rule",
    "선형회귀": lin,
    "랜덤포레스트": rf,
    "LightGBM": gbm,
    "LightGBM 보수적(q0.7)": lambda: gbm(objective="quantile", alpha=0.7),
}
ARR_MODELS = {"기준선(GBIS)": None, "규칙(그룹평균)": "rule", "선형회귀": lin,
              "랜덤포레스트": rf, "LightGBM": gbm}

seat_res, arr_res = [], []
for k in range(3, len(days)):                     # 최소 3일로 학습
    tr, te = df[df.day.isin(days[:k])], df[df.day == days[k]]
    X_tr, X_te = tr[CAT + NUM], te[CAT + NUM]
    for name, mk in SEAT_MODELS.items():          # 감소량을 학습 → 좌석 = 현재 - 감소량
        if mk is None:
            drop = np.zeros(len(te))
        elif mk == "rule":
            drop = rule_fit(tr, "drop")(te)
        else:
            drop = mk().fit(X_tr, tr["drop"]).predict(X_te)
        pred = np.clip(np.round(te.seat_now - drop), 0, te.seat_now)
        real, ps, rs = te.seat_at.values, stage(pred), stage(te.seat_at.values)
        seat_res.append(dict(model=name, day=days[k], n=len(te),
                             MAE=np.abs(pred - real).mean(),
                             단계정확도=(ps == rs).mean(),
                             만석인데OK=((rs == "RISK") & (ps == "OK")).sum() / max((rs == "RISK").sum(), 1),
                             여유인데RISK=((rs == "OK") & (ps == "RISK")).sum() / max((rs == "OK").sum(), 1)))
    for name, mk in ARR_MODELS.items():           # GBIS 오차(분)를 학습 → 도착 = GBIS + 오차
        if mk is None:
            corr = np.zeros(len(te))
        elif mk == "rule":
            corr = rule_fit(tr, "err_min")(te)
        else:
            corr = mk().fit(X_tr, tr["err_min"]).predict(X_te)
        arr_res.append(dict(model=name, day=days[k], MAE=np.abs(te.err_min.values - corr).mean()))

pd.set_option("display.width", 200)
s = pd.DataFrame(seat_res)
print(f"평일 {len(days)}일, 시험 {len(days) - 3}회 (4일째~마지막 날, 매번 그 전날까지로 학습)\n")
print("=== 좌석 (시험일 평균) ===")
print(s.groupby("model", sort=False)[["MAE", "단계정확도", "만석인데OK", "여유인데RISK"]].mean().round(3))
print("\n좌석 MAE 날짜별:")
print(s.pivot(index="day", columns="model", values="MAE")[list(SEAT_MODELS)].round(2))
a = pd.DataFrame(arr_res)
print("\n=== 도착 시각 MAE (분, 시험일 평균) ===")
print(a.groupby("model", sort=False).MAE.mean().round(3))

# 최종 LightGBM 특징 중요도 (전체 학습)
m = gbm().fit(df[CAT + NUM], df["drop"])
print("\n좌석 LightGBM 특징 중요도:", dict(zip(CAT + NUM, m.feature_importances_)))
