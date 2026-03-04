import os
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")

# ── Statsmodels ───────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

# ── pmdarima ──────────────────────────────────────────────────────────────────
try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False

# ─── PATH ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "sarima")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── KONFIGURASI ──────────────────────────────────────────────────────────────
CONFIG = {
    "rolling_window_days"  : None,
    "test_size"            : 0.15,
    "seasonal_period"      : 52,
    "min_train_factor"     : 2,
    "enforce_stationarity" : False,
    "enforce_invertibility": False,
}


# ─── HELPER PATH ──────────────────────────────────────────────────────────────
def _pkl_path(villa_name: str) -> str:
    safe = villa_name.replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"{safe}_sarima.pkl")

def _meta_path(villa_name: str) -> str:
    safe = villa_name.replace(" ", "_").lower()
    return os.path.join(MODEL_DIR, f"{safe}_meta.pkl")


# ─── HELPER METRICS ───────────────────────────────────────────────────────────
def _mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Gunakan WAPE (Weighted Absolute Percentage Error)
    # Lebih stabil untuk data yang ada angka 0-nya
    total_error = np.sum(np.abs(y_true - y_pred))
    total_actual = np.sum(np.abs(y_true))

    if total_actual == 0:
        return 0.0

    return float((total_error / total_actual) * 100)

def _mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(np.array(y_true), np.array(y_pred)))

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(np.array(y_true), np.array(y_pred))))

def _rating(mape_val: float) -> str:
    if mape_val < 5:  return "Sangat Baik ✨"
    if mape_val < 10: return "Baik 👍"
    if mape_val < 20: return "Cukup 👌"
    return "Perlu Revisi ⚠️"


# ─── FIT SARIMA ───────────────────────────────────────────────────────────────
def fit_sarima(series: pd.Series,
               villa_name: str,
               seasonal_period: int = None,
               test_size: float = None) -> dict:

    if not STATSMODELS_OK:
        return None

    if seasonal_period is None:
        seasonal_period = CONFIG["seasonal_period"]
    if test_size is None:
        test_size = CONFIG["test_size"]

    # ── 1. Resample ke mingguan ───────────────────────────────────
    s = series.resample('W').mean().dropna()

    # ── 2. Normalize ke 0-1 sebelum apapun ───────────────────────
    if s.max() > 1.5:
        s = s / 100.0

    n = len(s)

    # ── 3. Split train/test ───────────────────────────────────────
    n_test  = max(int(n * test_size), 4)
    n_train = n - n_test

    # ── 4. m adaptif ─────────────────────────────────────────────
    if n_train < 2 * seasonal_period:
        m_used = 4
        D_used = 1
        print(f"  ℹ️  [{villa_name}] Data {n} minggu (<104), pakai m=4 (quarterly)")
    else:
        m_used = seasonal_period
        D_used = 1
        print(f"  ℹ️  [{villa_name}] Data {n} minggu (>=104), pakai m=52 (annual)")

    # ── 5. Validasi minimum data ──────────────────────────────────
    min_train_needed = max(2 * m_used + 4, 12)
    if n_train < min_train_needed:
        print(f"  ⚠️  [{villa_name}] Data terlalu pendek: n_train={n_train}, "
              f"butuh minimal {min_train_needed} minggu (m={m_used}), skip.")
        return None

    train, test = s.iloc[:n_train], s.iloc[n_train:]

    print(f"\n{'='*60}")
    print(f"[{villa_name}] SARIMA Fitting")
    print(f"  Data range : {s.index[0].date()} → {s.index[-1].date()} (n={n})")
    print(f"  Train      : {train.index[0].date()} → {train.index[-1].date()} (n={len(train)})")
    print(f"  Test       : {test.index[0].date()} → {test.index[-1].date()} (n={len(test)})")
    print(f"  Skala data : {s.min():.4f} – {s.max():.4f} (0-1)")
    print(f"  m digunakan: {m_used}")
    print('  Mencari parameter SARIMA...')

    if not PMDARIMA_OK:
        print(f'  ⚠️  pmdarima tidak tersedia, gunakan order default.')
        order   = (1, 1, 1)
        s_order = (1, 1, 1, m_used)
    else:
        # ── OPTIMASI KECEPATAN ────────────────────────────────────
        # 1. max_p/max_q dikurangi dari 3→2 — search space lebih kecil
        # 2. max_P/max_Q dikurangi dari 2→1 — paling berpengaruh untuk m=52
        # 3. n_fits dibatasi via max_order
        # 4. random_state=42 — deterministik, sama dengan notebook
        # Estimasi waktu: ~3-5 menit → ~1-2 menit per villa
        auto_model = auto_arima(
            train,
            start_p=0, max_p=2,     # ← dikurangi dari 3 (hemat ~30% waktu)
            start_q=0, max_q=2,     # ← dikurangi dari 3
            d=None,    max_d=2,
            start_P=0, max_P=1,     # ← dikurangi dari 2 (paling hemat untuk m=52)
            start_Q=0, max_Q=1,     # ← dikurangi dari 2
            D=D_used,
            m=m_used,
            seasonal=True,
            information_criterion='aic',
            stepwise=True,          # stepwise HARUS True agar cepat
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            random_state=42,        # deterministik = sama dengan notebook
            n_jobs=1,               # hindari overhead multiprocessing di Streamlit
        )
        order   = auto_model.order
        s_order = auto_model.seasonal_order

    print(f"  ARIMA order    : {order}")
    print(f"  Seasonal order : {s_order}")

    # ── 7. Refit dengan SARIMAX ───────────────────────────────────
    results = SARIMAX(
        train,
        order=order,
        seasonal_order=s_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    aic = results.aic
    print(f"  AIC            : {aic:.2f}")

    # ── 8. Evaluasi test set ──────────────────────────────────────
    forecast = results.forecast(steps=len(test)).clip(0, 1)
    y_true   = test.values
    y_pred   = forecast.values

    mae  = _mae(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    mape = _mape(y_true, y_pred)

    print(f"\n  📊 Test Set: MAE={mae:.4f} | RMSE={rmse:.4f} | MAPE={mape:.2f}%")

    return {
        'villa'         : villa_name,
        'model'         : results,
        'train'         : train,
        'test'          : test,
        'forecast'      : forecast,
        'order'         : order,
        'seasonal_order': s_order,
        'aic'           : aic,
        'mae'           : mae,
        'rmse'          : rmse,
        'mape'          : mape,
        'series'        : s,
        'rating'        : _rating(mape),
    }


# ─── TRAIN ALL ────────────────────────────────────────────────────────────────
def train_all(df_occ: pd.DataFrame,
              force_retrain: bool = False) -> dict:
    results     = {}
    villa_names = df_occ["villa_name"].unique()

    for villa_name in villa_names:
        pkl  = _pkl_path(villa_name)
        meta = _meta_path(villa_name)

        if os.path.exists(pkl) and os.path.exists(meta) and not force_retrain:
            cached = joblib.load(meta)
            cached["status"] = "loaded_from_cache"
            results[villa_name] = cached
            print(f"[{villa_name}] ✅ Loaded from cache")
            continue

        sub    = df_occ[df_occ["villa_name"] == villa_name].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        series = sub.set_index("date")["occupancy_pct"].sort_index()

        window = CONFIG.get("rolling_window_days")
        if window and len(series) > window:
            series = series.iloc[-window:]

        res = fit_sarima(series, villa_name)

        if res is not None:
            joblib.dump(res["model"], pkl)
            meta_data = {k: v for k, v in res.items() if k not in ("model", "train", "test", "forecast", "series")}
            meta_data["status"]      = "trained"
            meta_data["data_end"]    = str(res["series"].index[-1].date())
            meta_data["n_train"]     = len(res["train"])
            meta_data["n_test"]      = len(res["test"])
            meta_data["model_path"]  = pkl
            meta_data["_series_end"] = res["series"].index[-1]
            joblib.dump({**meta_data, "_series": res["series"]}, meta)
            results[villa_name] = meta_data
        else:
            results[villa_name] = {"status": "error", "villa_name": villa_name}

    return results


# ─── TRAIN & SAVE (single villa) ─────────────────────────────────────────────
def train_and_save(villa_name: str,
                   series: pd.Series,
                   force_retrain: bool = False) -> dict:
    pkl  = _pkl_path(villa_name)
    meta = _meta_path(villa_name)

    if os.path.exists(pkl) and os.path.exists(meta) and not force_retrain:
        cached = joblib.load(meta)
        cached["status"] = "loaded_from_cache"
        return cached

    window = CONFIG.get("rolling_window_days")
    if window and len(series) > window:
        series = series.iloc[-window:]

    try:
        res = fit_sarima(series, villa_name)
    except Exception as e:
        import traceback
        return {"status": "error", "message": f"fit_sarima exception: {type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    if res is None:
        s_check = series.resample('W').mean().dropna()
        if s_check.max() > 1.5:
            s_check = s_check / 100.0
        n           = len(s_check)
        test_size   = CONFIG["test_size"]
        s_period    = CONFIG["seasonal_period"]
        n_test      = max(int(n * test_size), 4)
        n_train     = n - n_test
        m_used      = 4 if n_train < 2 * s_period else s_period
        min_needed  = max(2 * m_used + 4, 12)
        if not STATSMODELS_OK:
            msg = "statsmodels tidak terinstall"
        elif n_train < min_needed:
            msg = f"Data terlalu pendek: {n} minggu total, n_train={n_train}, butuh minimal {min_needed} minggu (pakai m={m_used})"
        else:
            msg = "fit_sarima gagal — cek terminal/log untuk detail error"
        return {"status": "error", "message": msg}

    joblib.dump(res["model"], pkl)
    meta_data = {k: v for k, v in res.items() if k not in ("model", "train", "test", "forecast")}
    meta_data["status"]     = "trained"
    meta_data["data_end"]   = str(res["series"].index[-1].date())
    meta_data["n_train"]    = len(res["train"])
    meta_data["n_test"]     = len(res["test"])
    meta_data["model_path"] = pkl
    joblib.dump({**meta_data, "_series": res["series"]}, meta)
    return meta_data


# ─── FORECAST ─────────────────────────────────────────────────────────────────
def forecast(villa_name: str,
             horizon: int = 26,
             target_end_date: str = "2026-06-30") -> pd.DataFrame:
    pkl  = _pkl_path(villa_name)
    meta = _meta_path(villa_name)

    if not os.path.exists(pkl):
        return pd.DataFrame({"error": ["Model belum ditraining."]})

    try:
        model       = joblib.load(pkl)
        meta_data   = joblib.load(meta) if os.path.exists(meta) else {}
        full_series = meta_data.get("_series")

        if full_series is None:
            return pd.DataFrame({"error": ["Series tidak ditemukan di metadata."]})

        last_date   = full_series.index.max()
        target_date = pd.Timestamp(target_end_date)

        weeks_difference = (target_date - last_date).days / 7
        weeks_to_predict = int(np.ceil(weeks_difference))
        if weeks_to_predict < horizon:
            weeks_to_predict = horizon

        forecast_obj  = model.get_forecast(steps=weeks_to_predict)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci   = forecast_obj.conf_int()

        forecast_dates = pd.date_range(
            start=last_date + timedelta(weeks=1),
            periods=weeks_to_predict,
            freq='W'
        )

        forecast_df = pd.DataFrame({
            'predicted_occupancy': forecast_mean.values.clip(0, 1),
            'lower_bound'        : forecast_ci.iloc[:, 0].values.clip(0, 1),
            'upper_bound'        : forecast_ci.iloc[:, 1].values.clip(0, 1),
        }, index=forecast_dates)

        target_year  = target_date.year
        target_month = target_date.month
        mask = (
            (forecast_df.index.year == target_year) &
            (forecast_df.index.month >= 1) &
            (forecast_df.index.month <= target_month)
        )
        forecast_2026 = forecast_df[mask].copy()

        if len(forecast_2026) == 0:
            return pd.DataFrame({"error": ["Tidak ada data untuk range tanggal yang diminta."]})

        forecast_2026["week"]      = [f"W{i+1}" for i in range(len(forecast_2026))]
        forecast_2026["month"]     = forecast_2026.index.strftime("%b %Y")
        forecast_2026["month_num"] = forecast_2026.index.month
        forecast_2026.index.name   = "date"

        return forecast_2026

    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


# ─── PREDICT 2026 ─────────────────────────────────────────────────────────────
def predict_2026(sarima_result: dict, villa_name: str) -> dict:
    if sarima_result is None or 'model' not in sarima_result:
        print(f"❌ Error: sarima_result invalid untuk {villa_name}")
        return None

    model       = sarima_result['model']
    full_series = sarima_result.get('series')

    if full_series is None:
        print(f"❌ Error: Data series tidak ditemukan untuk {villa_name}")
        return None

    last_date   = full_series.index.max()
    target_date = pd.Timestamp('2026-06-30')

    weeks_difference = (target_date - last_date).days / 7
    weeks_to_predict = int(np.ceil(weeks_difference))
    if weeks_to_predict < 26:
        weeks_to_predict = 26

    print(f"\n{'='*70}")
    print(f"PREDIKSI 2026 (Jan-Jun): {villa_name}")
    print(f"  Data terakhir      : {last_date.date()}")
    print(f"  Minggu diprediksi  : {weeks_to_predict}")

    try:
        forecast_obj  = model.get_forecast(steps=weeks_to_predict)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci   = forecast_obj.conf_int()
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return None

    forecast_dates = pd.date_range(
        start=last_date + timedelta(weeks=1),
        periods=weeks_to_predict,
        freq='W'
    )

    forecast_df = pd.DataFrame({
        'predicted_occupancy': forecast_mean.values.clip(0, 1),
        'lower_bound'        : forecast_ci.iloc[:, 0].values.clip(0, 1),
        'upper_bound'        : forecast_ci.iloc[:, 1].values.clip(0, 1),
    }, index=forecast_dates)

    mask_2026 = (
        (forecast_df.index.year == 2026) &
        (forecast_df.index.month >= 1) &
        (forecast_df.index.month <= 6)
    )
    forecast_2026 = forecast_df[mask_2026].copy()

    if len(forecast_2026) == 0:
        print("⚠️ WARNING: Tidak ada data 2026 yang ter-generate!")
        return None

    forecast_2026_copy               = forecast_2026.copy()
    forecast_2026_copy['month']      = forecast_2026_copy.index.month
    forecast_2026_copy['month_name'] = forecast_2026_copy.index.strftime('%B')
    monthly_avg = forecast_2026_copy.groupby(
        ['month', 'month_name']
    )['predicted_occupancy'].mean()

    print(f"📊 Rata-rata Jan-Jun 2026: {forecast_2026['predicted_occupancy'].mean()*100:.2f}%")
    for (mn, mname), avg in monthly_avg.items():
        print(f"   {mname:10s}: {avg*100:6.2f}%")

    return {
        'villa'        : villa_name,
        'forecast_full': forecast_df,
        'forecast_2026': forecast_2026,
        'monthly_avg'  : monthly_avg,
        'full_series'  : full_series,
        'model'        : model,
    }


# ─── PREDICT ALL 2026 ─────────────────────────────────────────────────────────
def predict_all_2026(sarima_results: dict) -> dict:
    predictions_2026 = {}
    success_count    = 0
    failed_villas    = []

    print("\n" + "="*80)
    print("🚀 MULAI PROSES PREDIKSI 2026 UNTUK SEMUA VILLA")
    print("="*80)

    for (area, villa), res in sarima_results.items():
        if res is not None:
            try:
                pred = predict_2026(res, f"{area} - {villa}")
                if pred is not None:
                    predictions_2026[(area, villa)] = pred
                    success_count += 1
                else:
                    failed_villas.append(f"{area} - {villa}")
            except Exception as e:
                print(f"❌ Exception untuk {area} - {villa}: {e}")
                failed_villas.append(f"{area} - {villa}")

    print(f"\n✅ Berhasil: {success_count} villa")
    if failed_villas:
        print(f"⚠️ Gagal: {failed_villas}")

    return predictions_2026


# ─── GET META ─────────────────────────────────────────────────────────────────
def get_meta(villa_name: str) -> dict:
    meta_p = _meta_path(villa_name)
    if os.path.exists(meta_p):
        m = joblib.load(meta_p)
        return {k: v for k, v in m.items() if k not in ("_series",)}
    return {}


def get_all_meta(villa_names: list) -> pd.DataFrame:
    rows = []
    for name in villa_names:
        meta = get_meta(name)
        if meta:
            rows.append({
                "Vila"          : name,
                "Order"         : str(meta.get("order", "-")),
                "Seasonal Order": str(meta.get("seasonal_order", "-")),
                "AIC"           : meta.get("aic", "-"),
                "MAE"           : meta.get("mae", "-"),
                "RMSE"          : meta.get("rmse", "-"),
                "MAPE (%)"      : meta.get("mape", "-"),
                "Rating"        : meta.get("rating", "-"),
                "Status"        : meta.get("status", "-"),
                "Data Hingga"   : meta.get("data_end", "-"),
            })
        else:
            rows.append({
                "Vila"          : name,
                "Order"         : "-",
                "Seasonal Order": "-",
                "AIC"           : "-",
                "MAE"           : "-",
                "RMSE"          : "-",
                "MAPE (%)"      : "-",
                "Rating"        : "⏳ Belum ditraining",
                "Status"        : "not_trained",
                "Data Hingga"   : "-",
            })
    return pd.DataFrame(rows)


def model_exists(villa_name: str) -> bool:
    return os.path.exists(_pkl_path(villa_name))

def models_status(villa_names: list) -> dict:
    return {
        name: "trained" if model_exists(name) else "not_trained"
        for name in villa_names
    }

def engine_info() -> dict:
    return {
        "statsmodels_ok"  : STATSMODELS_OK,
        "pmdarima_ok"     : PMDARIMA_OK,
        "data_frequency"  : "weekly",
        "seasonal_period" : CONFIG["seasonal_period"],
        "random_state"    : 42,
        "max_p_q"         : 2,
        "max_P_Q"         : 1,
        "config"          : CONFIG,
    }