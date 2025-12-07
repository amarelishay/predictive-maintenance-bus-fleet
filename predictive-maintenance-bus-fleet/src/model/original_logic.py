# =========================================================
# Predictive Maintenance – Dual Horizons (7d & 30d) — Final Fixed
# - Strict future next-failure per row (בטוח, בלי IndexError)
# - Rolling('30D') אמיתי ל-fail_count_30d (חוזר כסדרה מיושרת)
# - EWM מהיר עם groupby().transform
# - Z-Score פר-אוטובוס להסברים קלים + likely fault
# - days_since_last_fail מחושב ונכנס לפיצ'רים
# - בניית מטריצת פיצ'רים חסינה (דלוג על עמודות חסרות)
# - כתיבה ל-DB + אינדקסים + AUPRC ו-Sanity Prints
# =========================================================
import os, re, warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation
import joblib

warnings.filterwarnings("ignore")

# -------------------- CONFIG --------------------
HORIZONS = (7, 30)                # אופקי חיזוי
RECALL_TARGET = 0.60              # יעד Recall לסף
TEST_FRACTION_BY_TIME = 0.20      # פיצול זמן ל-test (אחרון)
RANDOM_SEED = 42
MODELS_DIR = "models"
CHUNKSIZE_TO_SQL = 20_000
REUSE_SAVED_THRESHOLDS = True     # מחזר ספים אם קיימים

load_dotenv()
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB   = os.getenv("PG_DB",   "hacketon")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASSWORD", "1234")
ENGINE  = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------- HELPERS --------------------
def sanitize_feature_names(columns: list[str]) -> tuple[list[str], dict]:
    used, mapping, safe_cols = set(), {}, []
    for c in columns:
        s = re.sub(r"[^A-Za-z0-9_]", "_", c)
        base = s
        k = 1
        while s in used:
            k += 1
            s = f"{base}_{k}"
        used.add(s)
        mapping[c] = s
        safe_cols.append(s)
    return safe_cols, mapping

def choose_threshold(y_true, y_scores, recall_target=0.6):
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    mask = rec[:-1] >= recall_target
    if mask.any():
        i = np.argmax(prec[:-1][mask])
        return float(thr[mask][i]), f"Recomputed (Max Precision with Recall>={recall_target})"
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[np.argmax(f1)]), "Recomputed (Max F1; no point met Recall target)"

def time_split_by_quantile(dates: pd.Series, test_fraction: float):
    cut = dates.quantile(1 - test_fraction)
    tr_idx = dates <= cut
    te_idx = dates >  cut
    return tr_idx.values, te_idx.values

def load_saved_threshold(h: int):
    p = os.path.join(MODELS_DIR, f"h{h}", "threshold.txt")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                val = float(f.read().strip())
                print(f"[h{h}] Trying to load threshold from: {p} | content='{val}'")
                return val
        except Exception:
            return None
    return None

def infer_fault_from_features(top_feats: list[str], part_primary) -> str:
    name = str(part_primary if not pd.isna(part_primary) else "").lower()
    if "brake" in name:        return "Brake"
    if "engine" in name:       return "Engine"
    if "radiator" in name or "cool" in name: return "Cooling"
    if "ac" in name:           return "AC"
    if "trans" in name:        return "Transmission"
    if "susp" in name:         return "Suspension"
    if "elect" in name:        return "Electrical"

    t = " ".join(top_feats).lower()
    if any(k in t for k in ["temp_delta", "temp_std", "temp_ewm", "temperature_avg_c"]):
        return "Cooling/Engine"
    if any(k in t for k in ["speed_delta", "speed_std"]):
        return "Transmission/Suspension"
    if "engine_growth" in t:
        return "Engine"
    if any(k in t for k in ["part_wear_pct", "part_km_since_event"]):
        return "Part wear"
    return "Unknown"

def light_explanations_with_fault_per_bus(
    X_all: pd.DataFrame, base_df_for_part: pd.DataFrame, bus_ids: pd.Series
) -> tuple[pd.Series, pd.Series]:
    exp_cols = [
        "part_wear_pct","part_km_since_event","temp_delta","speed_delta",
        "temp_std_7d","speed_std_7d","engine_growth","mileage_growth"
    ]
    use = [c for c in exp_cols if c in X_all.columns]
    if not use:
        n = len(X_all)
        return pd.Series(["no standout factors"]*n), pd.Series(["Unknown"]*n)

    # Z-score per bus
    zs = X_all[use].copy()
    zs = zs.groupby(bus_ids).transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    zs = zs.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    reasons, likely_faults = [], []
    for i in range(len(zs)):
        top = zs.iloc[i].sort_values(ascending=False).head(3)
        top_feats = [k for k, v in top.items() if np.isfinite(v) and v > 0]
        txt  = ", ".join([f"{k}↑ ({top[k]:.1f}σ)" for k in top_feats]) if top_feats else "no standout factors"
        reasons.append(txt)
        likely_faults.append(infer_fault_from_features(top_feats, base_df_for_part.iloc[i]["part_primary"]))
    return pd.Series(reasons), pd.Series(likely_faults)

# -------- Strict-future next failure (בטוח, בלי IndexError) --------
def compute_next_failure_strict_future(g: pd.DataFrame) -> pd.Series:
    fails = np.sort(g.loc[g["had_failure"].eq(1), "date"].to_numpy(dtype="datetime64[ns]"))
    out = np.empty(len(g), dtype="datetime64[ns]")
    out[:] = np.datetime64("NaT")
    if fails.size == 0:
        return pd.Series(pd.to_datetime(out), index=g.index)

    idx = np.searchsorted(fails, g["date"].to_numpy(dtype="datetime64[ns]"), side="right")
    valid = idx < fails.size
    if np.any(valid):
        out[valid] = fails[idx[valid]]
    return pd.Series(pd.to_datetime(out), index=g.index)

# -------------------- EXTRACT --------------------
parts_clause = """
LEFT JOIN public.bridge_fault_part bfp ON bfp.fault_id = f.fault_id
LEFT JOIN public.dim_part p_bridge     ON p_bridge.part_id = bfp.part_id
LEFT JOIN LATERAL (
  SELECT p2.part_name, p2.expected_lifetime_km
  FROM public.dim_part p2
  WHERE
    (f.failure_type ILIKE '%Brake%'       AND p2.part_name ILIKE '%Brake%') OR
    (f.failure_type ILIKE '%Engine%'      AND (p2.part_name ILIKE '%Engine%' OR p2.part_name ILIKE '%Oil%' OR p2.part_name ILIKE '%Filter%')) OR
    (f.failure_type ILIKE '%Electrical%'  AND p2.part_name ILIKE '%Elect%') OR
    (f.failure_type ILIKE '%Cooling%'     AND (p2.part_name ILIKE '%Radiator%' OR p2.part_name ILIKE '%Cool%')) OR
    (f.failure_type ILIKE '%Transmission%'AND p2.part_name ILIKE '%Trans%') OR
    (f.failure_type ILIKE '%Suspension%'  AND p2.part_name ILIKE '%Suspens%')
  ORDER BY p2.expected_lifetime_km NULLS LAST
  LIMIT 1
) p_kw ON TRUE
"""

SQL = f"""
SELECT
  b.bus_id,
  s.date_id::date AS date,
  b.region_types,
  d.season,
  s.trip_distance_km, s.avg_speed_kmh, s.passengers_avg,
  s.temperature_avg_c, s.engine_hours_total, s.mileage_total_km,
  COALESCE(s.failure_flag,FALSE)     AS failure_flag,
  COALESCE(s.maintenance_flag,FALSE) AS maintenance_flag,
  f.failure_type,
  COALESCE(p_bridge.part_name, p_kw.part_name) AS part_name,
  COALESCE(p_bridge.expected_lifetime_km, p_kw.expected_lifetime_km) AS expected_lifetime_km
FROM public.fact_bus_status_star s
JOIN public.dim_bus_star b ON b.bus_sk = s.bus_sk
JOIN public.dim_date d      ON d.date_id = s.date_id
LEFT JOIN public.dim_fault f ON f.fault_id = s.fault_id
{parts_clause}
"""

with ENGINE.connect() as con:
    df_raw = pd.read_sql_query(text(SQL), con, parse_dates=["date"])

# -------------------- FEATURE ENGINEERING --------------------
grp = df_raw.groupby(["bus_id", "date"], as_index=False)
def first(s): return s.iloc[0]
df = grp.agg({
    "region_types": first, "season": first,
    "trip_distance_km": first, "avg_speed_kmh": first, "passengers_avg": first,
    "temperature_avg_c": first, "engine_hours_total": first, "mileage_total_km": first,
    "failure_flag": "max", "maintenance_flag": "max",
    "failure_type": first,
    "expected_lifetime_km": lambda s: np.nan if s.dropna().empty else float(np.nanmin(s)),
    "part_name": lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan
}).rename(columns={"expected_lifetime_km":"expected_lifetime_min","part_name":"part_primary"}).copy()

df_base_for_explain = df[["bus_id","date","part_primary"]].copy()
df = df.sort_values(["bus_id","date"]).reset_index(drop=True)

# --------- LABELS: strict future next failure ---------
df["had_failure"] = df["failure_flag"].astype(int)
df["next_failure_date"] = (
    df.groupby("bus_id", group_keys=False)
      .apply(compute_next_failure_strict_future)
)
days_to_next = (df["next_failure_date"] - df["date"]).dt.days.fillna(999)
df["failure_next_7d"]  = (days_to_next <= 7).astype(int)
df["failure_next_30d"] = (days_to_next <= 30).astype(int)

# days_since_last_fail
tmp_fail_date = df["date"].where(df["had_failure"].eq(1))
df["last_fail_date"] = tmp_fail_date.groupby(df["bus_id"]).ffill()
df["days_since_last_fail"] = (df["date"] - df["last_fail_date"]).dt.days.fillna(999).astype(float)
df.drop(columns=["last_fail_date"], inplace=True)

# גלגולים/STD/דלתא
for col in ["temperature_avg_c", "avg_speed_kmh"]:
    g = df.groupby("bus_id")[col]
    nm = "temp" if "temp" in col else "speed"
    df[f"avg_{nm}_3d"] = g.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f"{nm}_delta"]  = g.diff().fillna(0)
    df[f"{nm}_std_7d"] = g.rolling(7, min_periods=3).std().reset_index(level=0, drop=True).fillna(0)

# דינמיקה ושימוש חלק
df["engine_growth"]  = df.groupby("bus_id")["engine_hours_total"].diff().fillna(0)
df["mileage_growth"] = df.groupby("bus_id")["mileage_total_km"].diff().fillna(0)
df["daily_km"]       = df.groupby("bus_id")["mileage_total_km"].diff().clip(lower=0).fillna(0)

reset_mask  = df["maintenance_flag"].fillna(False) | df["failure_flag"].fillna(False)
part_change = (df["part_primary"] != df.groupby("bus_id")["part_primary"].shift(1)).fillna(True)
seg_break   = reset_mask | part_change
df["segment_id"] = seg_break.groupby(df["bus_id"]).cumsum()

df["part_km_since_event"]   = df.groupby(["bus_id","segment_id"])["daily_km"].cumsum().fillna(0)
df["expected_lifetime_min"] = df["expected_lifetime_min"].astype(float)
df["part_wear_pct"]         = (df["part_km_since_event"]/df["expected_lifetime_min"]).replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(0,2.0)
df["part_remaining_km"]     = (df["expected_lifetime_min"] - df["part_km_since_event"]).clip(-1e6,1e6).fillna(0.0)
df["days_since_part_event"] = df.groupby(["bus_id","segment_id"])["date"].transform(lambda s: (s - s.min()).dt.days).fillna(0)

# היסטוריית תקלות – 30 יום אמיתי (Series מיושר)
def fail_count_30d_per_group(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("date")
    s = pd.Series(g["had_failure"].values, index=pd.to_datetime(g["date"].values))
    r = s.rolling("30D", min_periods=1).sum()
    r.index = g.index  # ליישר לאינדקס המקורי של df
    return r

df["fail_count_30d"] = (
    df.groupby("bus_id", group_keys=False)
      .apply(fail_count_30d_per_group)
      .astype(float)
)

# דמיז
df = pd.get_dummies(df, columns=["region_types","season"], drop_first=True, sparse=False)

# אותות ארוכים (transform במקום apply)
for col, name in [("temperature_avg_c","temp"), ("avg_speed_kmh","speed")]:
    g = df.groupby("bus_id")[col]
    df[f"{name}_roll_mean_14"] = g.rolling(14, min_periods=5).mean().reset_index(level=0, drop=True)
    df[f"{name}_std_14"]       = g.rolling(14, min_periods=5).std().reset_index(level=0, drop=True).fillna(0)
    df[f"{name}_ewm_7"]        = g.transform(lambda s: s.ewm(span=7, adjust=False).mean())
    df[f"{name}_lag1"]         = g.shift(1).bfill()

df["temp_x_speed"]      = df["temperature_avg_c"] * df["avg_speed_kmh"]
df["temp_x_passengers"] = df["temperature_avg_c"] * df["passengers_avg"]

df["wear_index"]        = (df["mileage_total_km"]/df["expected_lifetime_min"]).replace([np.inf,-np.inf],np.nan).fillna(0.0).clip(0,5.0)
df["remaining_life_km"] = (df["expected_lifetime_min"] - df["mileage_total_km"]).fillna(0.0).clip(-1e6,1e6)

top_parts = df_base_for_explain["part_primary"].dropna().value_counts().head(8).index.tolist()
df_base_for_explain["part_primary"] = df_base_for_explain["part_primary"].where(
    df_base_for_explain["part_primary"].isin(top_parts), other="OtherNA"
)
df["part_primary"] = df_base_for_explain["part_primary"]
df = pd.get_dummies(df, columns=["part_primary"], drop_first=True, sparse=False)

# -------------------- FEATURES (חסין לעמודות חסרות) --------------------
base_features = [
    "trip_distance_km","avg_speed_kmh","passengers_avg",
    "temperature_avg_c","engine_hours_total","mileage_total_km",
    "avg_temp_3d","avg_speed_3d","engine_growth"
]
trend_features = [
    "mileage_growth","temp_delta","speed_delta",
    "temp_std_7d","speed_std_7d",
    "days_since_last_fail","fail_count_30d",
    "temp_roll_mean_14","speed_roll_mean_14",
    "temp_std_14","speed_std_14",
    "temp_ewm_7","speed_ewm_7",
    "temp_lag1","speed_lag1",
    "temp_x_speed","temp_x_passengers",
]
part_usage_features = ["part_km_since_event","part_wear_pct","part_remaining_km","days_since_part_event"]
region_season_dummies = [c for c in df.columns if c.startswith("region_types_") or c.startswith("season_")]
part_dummies = [c for c in df.columns if c.startswith("part_primary_")]
wear_features = ["expected_lifetime_min","wear_index","remaining_life_km"]

FEATURES_RAW_ALL = (
    base_features + trend_features + wear_features +
    part_usage_features + region_season_dummies + part_dummies
)
available_features = [c for c in FEATURES_RAW_ALL if c in df.columns]
missing_features = sorted(set(FEATURES_RAW_ALL) - set(available_features))
if missing_features:
    print("⚠️ Missing features skipped:", missing_features)

X_all = df[available_features].copy().fillna(0).astype(np.float32)
safe_cols, name_map = sanitize_feature_names(available_features)
X_all.columns = safe_cols
dates = df["date"]

# drop constant / near-constant / duplicate columns
const_cols = [c for c in X_all.columns if X_all[c].nunique(dropna=False) <= 1]
if const_cols:
    X_all = X_all.drop(columns=const_cols)

near_const = []
for c in X_all.columns:
    vc = X_all[c].value_counts(normalize=True, dropna=False)
    if len(vc) and vc.iloc[0] > 0.995:
        near_const.append(c)
if near_const:
    X_all = X_all.drop(columns=near_const)

X_all = X_all.loc[:, ~X_all.T.duplicated()]

# -------------------- TRAIN PER HORIZON --------------------
def train_and_predict_for_horizon(h: int):
    label_col = f"failure_next_{h}d"
    y_all = df[label_col].astype(int).values

    tr_idx, te_idx = time_split_by_quantile(dates, TEST_FRACTION_BY_TIME)

    pos_rate = float(y_all.mean())
    spw = (1 - pos_rate) / max(pos_rate, 1e-6)

    model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        force_col_wise=True,
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        min_gain_to_split=1e-3,
        colsample_bytree=0.8,
        subsample=0.8,
        scale_pos_weight=spw,
        verbosity=-1
    )

    dates_tr = dates[tr_idx]
    cut_val  = dates_tr.quantile(0.85)
    val_mask = (dates_tr > cut_val).values
    tr_mask  = (dates_tr <= cut_val).values

    X_tr, y_tr = X_all.loc[tr_idx].iloc[tr_mask], y_all[tr_idx][tr_mask]
    X_va, y_va = X_all.loc[tr_idx].iloc[val_mask], y_all[tr_idx][val_mask]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False),
                   log_evaluation(0)]
    )

    saved_thr = load_saved_threshold(h) if REUSE_SAVED_THRESHOLDS else None
    y_scores_test = model.predict_proba(X_all.loc[te_idx])[:, 1]

    def recompute_and_save():
        thr, crit = choose_threshold(y_all[te_idx], y_scores_test, RECALL_TARGET)
        os.makedirs(os.path.join(MODELS_DIR, f"h{h}"), exist_ok=True)
        with open(os.path.join(MODELS_DIR, f"h{h}", "threshold.txt"), "w", encoding="utf-8") as f:
            f.write(str(thr))
        return thr, crit

    if saved_thr is None:
        thr, crit = recompute_and_save()
    else:
        thr, crit = float(saved_thr), "Loaded saved threshold"
        preds_tmp = (y_scores_test >= thr).astype(int)
        pos_in_test = int(preds_tmp.sum())
        if pos_in_test == 0 or pos_in_test == len(preds_tmp):
            print(f"[h{h}] ⚠️ Degenerate predictions with threshold={thr:.3f} "
                  f"(pos={pos_in_test}/{len(preds_tmp)}; rate={100*pos_in_test/len(preds_tmp):.3f}%) → recomputing.")
            thr, crit = recompute_and_save()

    y_pred = (y_scores_test >= thr).astype(int)

    print(f"\n====== HORIZON {h} (Auto-Optimized) ======")
    print(f"Train size: {tr_idx.sum():,} | Test size: {te_idx.sum():,} | pos_rate(all)={pos_rate:.3f}")
    print(f"Chosen threshold: {thr:.3f} | {crit}")
    print("=== Classification report ===")
    print(classification_report(y_all[te_idx], y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_all[te_idx], y_pred))
    ap = average_precision_score(y_all[te_idx], y_scores_test)
    print(f"AUPRC (h={h}): {ap:.3f}")

    name_rev = {v: k for k, v in name_map.items()}
    used_cols = X_all.columns
    imp = pd.Series(model.feature_importances_, index=used_cols).sort_values(ascending=False).reset_index()
    imp.columns = ["feature_sanitized", "importance"]
    imp["feature"] = imp["feature_sanitized"].map(lambda s: name_rev.get(s, s))
    imp = imp[["feature", "importance"]]

    with ENGINE.begin() as con:
        imp.to_sql(f"feature_importance_global_h{h}", con=con, schema="public",
                   if_exists="replace", index=False, chunksize=CHUNKSIZE_TO_SQL, method="multi")
    print(f"✅ feature_importance_global_h{h} נכתבה ל-DB")

    hdir = os.path.join(MODELS_DIR, f"h{h}")
    os.makedirs(hdir, exist_ok=True)
    joblib.dump(model, f"{hdir}/model.pkl")
    joblib.dump(list(used_cols), f"{hdir}/features.pkl")

    proba_all = model.predict_proba(X_all)[:, 1]
    label_all = (proba_all >= thr).astype(int)

    return {"h": h, "thr": thr, "proba": proba_all, "label": label_all}

results = {h: train_and_predict_for_horizon(h) for h in HORIZONS}

# -------------------- BUILD OUTPUT TABLES --------------------
# sanity
corr_labels = df["failure_next_7d"].corr(df["failure_next_30d"])
print("✅ label correlation (7d vs 30d):", corr_labels)
print("✅ probs equal? ", np.allclose(results[7]["proba"], results[30]["proba"]))

# הסברים קלים + likely fault (Z-score per bus)
X_for_explain = df[available_features].copy().fillna(0)  # להשתמש באותן עמודות זמינות
reasons, likely_faults = light_explanations_with_fault_per_bus(
    X_for_explain, df_base_for_explain, df["bus_id"]
)

# BI table
bi = df_base_for_explain[["bus_id","date"]].copy()
for h in HORIZONS:
    bi[f"failure_next_{h}d"] = df[f"failure_next_{h}d"].astype(int)
    bi[f"proba_{h}d"] = results[h]["proba"]
    bi[f"label_{h}d"] = results[h]["label"]

bi["failure_reason"] = reasons
bi["likely_fault"]   = likely_faults

with ENGINE.begin() as con:
    bi.sort_values(["bus_id","date"]).to_sql(
        "predictions_for_powerbi", con=con, schema="public",
        if_exists="replace", index=False, chunksize=CHUNKSIZE_TO_SQL, method="multi"
    )

# current status per bus
current_cols = ["bus_id","date"] + [f"proba_{h}d" for h in HORIZONS] + [f"label_{h}d" for h in HORIZONS]
ml_current = (
    bi[current_cols]
      .sort_values(["bus_id","date"])
      .groupby("bus_id", as_index=False)
      .tail(1)
      .rename(columns={"date":"last_date"})
)

with ENGINE.begin() as con:
    ml_current.to_sql("ml_current_risk", con=con, schema="public",
                      if_exists="replace", index=False, chunksize=CHUNKSIZE_TO_SQL, method="multi")

# אינדקסים מומלצים
with ENGINE.begin() as con:
    con.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_predictions_bus_date ON public.predictions_for_powerbi (bus_id, date);
        CREATE INDEX IF NOT EXISTS ix_predictions_date ON public.predictions_for_powerbi (date);
        CREATE INDEX IF NOT EXISTS ix_ml_current_risk_bus ON public.ml_current_risk (bus_id);
        CREATE INDEX IF NOT EXISTS ix_fi_h7_feature ON public.feature_importance_global_h7 (feature);
        CREATE INDEX IF NOT EXISTS ix_fi_h30_feature ON public.feature_importance_global_h30 (feature);
    """))

print("✅ predictions_for_powerbi + ml_current_risk נכתבו ל-Postgres")
print("✅ Done.")
