import os
import joblib
import warnings
import numpy as np
import pandas as pd
import io
from database import run_query, get_conn
from datetime import timedelta

warnings.filterwarnings("ignore")

# ── Statsmodels ───────────────────────────────────────────────────────────────
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf
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
    "seasonal_period"      : 52,   # batas atas m — nilai aktual ditentukan otomatis
    "min_train_factor"     : 2,
    "enforce_stationarity" : False,
    "enforce_invertibility": False,
    # ── Sanity check threshold ─────────────────────────────────────────────────
    # Forecast dianggap tidak wajar jika:
    #   - >70% nilai prediksi berada di >= 99% (ceiling)
    #   - >70% nilai prediksi berada di <= 1%  (floor)
    #   - std deviasi < 0.02 (flat / tidak bergerak)
    "sanity_flat_threshold"   : 0.70,   # proporsi nilai ekstrem
    "sanity_std_threshold"    : 0.02,   # std minimum
    # ── Ceiling data detection ─────────────────────────────────────────────────
    # Data villa dianggap "ceiling-dominated" jika >60% hari bernilai 100%
    "ceiling_data_threshold"  : 0.60,
    # ── CI width untuk fallback ────────────────────────────────────────────────
    "fallback_ci_half_width"  : 0.08,
}

# ─── CACHE model_exists ───────────────────────────────────────────────────────
_model_exists_cache: dict = {}


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
    total_error  = np.sum(np.abs(y_true - y_pred))
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


# ─── [BARU] DETEKSI DATA CEILING ──────────────────────────────────────────────
def _detect_ceiling_data(series_normalized: pd.Series,
                          villa_name: str = "") -> dict:
    """
    Deteksi apakah data training didominasi nilai ceiling (mendekati 100%).

    Ini penting karena SARIMA tidak bisa belajar pola variasi
    jika mayoritas data sudah berada di batas atas (1.0 dalam skala 0-1).

    Returns dict dengan:
      - is_ceiling_dominated: bool
      - pct_at_ceiling: float (proporsi nilai >= 0.95)
      - pct_at_floor:   float (proporsi nilai <= 0.05)
      - mean_val:       float
      - std_val:        float
    """
    pct_ceiling = float((series_normalized >= 0.95).mean())
    pct_floor   = float((series_normalized <= 0.05).mean())
    mean_val    = float(series_normalized.mean())
    std_val     = float(series_normalized.std())
    threshold   = CONFIG["ceiling_data_threshold"]

    is_ceiling  = pct_ceiling > threshold
    is_floor    = pct_floor   > threshold
    is_low_var  = std_val < 0.05  # hampir tidak ada variasi sama sekali

    print(f"\n  [Ceiling Check — {villa_name}]")
    print(f"    % nilai >= 95% : {pct_ceiling:.1%}  (threshold={threshold:.0%})")
    print(f"    % nilai <= 5%  : {pct_floor:.1%}")
    print(f"    Mean           : {mean_val:.4f}  |  Std: {std_val:.4f}")
    print(f"    Ceiling dominated: {'⚠️  YA' if is_ceiling else '✅ tidak'}")
    print(f"    Floor dominated  : {'⚠️  YA' if is_floor   else '✅ tidak'}")
    print(f"    Low variance     : {'⚠️  YA' if is_low_var else '✅ tidak'}")

    return {
        "is_ceiling_dominated": is_ceiling,
        "is_floor_dominated"  : is_floor,
        "is_low_variance"     : is_low_var,
        "pct_at_ceiling"      : pct_ceiling,
        "pct_at_floor"        : pct_floor,
        "mean_val"            : mean_val,
        "std_val"             : std_val,
    }


# ─── [BARU] SANITY CHECK FORECAST ────────────────────────────────────────────
def _sanity_check_forecast(forecast_series: pd.Series,
                            villa_name: str = "") -> dict:
    """
    Periksa apakah hasil forecast wajar atau tidak.

    Forecast dianggap tidak wajar jika:
    1. Terlalu banyak nilai menempel di ceiling (>=99%)
    2. Terlalu banyak nilai menempel di floor (<=1%)
    3. Hampir tidak ada variasi (std < threshold) — forecast flat

    Returns dict dengan:
      - is_sane:       bool  (True = wajar, False = perlu fallback)
      - reason:        str   (penjelasan kenapa tidak wajar)
      - pct_at_ceiling float
      - pct_at_floor:  float
      - std_val:       float
    """
    flat_thr = CONFIG["sanity_flat_threshold"]
    std_thr  = CONFIG["sanity_std_threshold"]

    pct_ceiling = float((forecast_series >= 0.99).mean())
    pct_floor   = float((forecast_series <= 0.01).mean())
    std_val     = float(forecast_series.std())

    reason = ""
    is_sane = True

    if pct_ceiling > flat_thr:
        is_sane = False
        reason  = f"Forecast flat di ceiling: {pct_ceiling:.0%} nilai >= 99%"
    elif pct_floor > flat_thr:
        is_sane = False
        reason  = f"Forecast flat di floor: {pct_floor:.0%} nilai <= 1%"
    elif std_val < std_thr:
        is_sane = False
        reason  = f"Forecast tidak bergerak: std={std_val:.4f} < {std_thr}"

    print(f"\n  [Sanity Check — {villa_name}]")
    print(f"    % nilai ceiling : {pct_ceiling:.1%}")
    print(f"    % nilai floor   : {pct_floor:.1%}")
    print(f"    Std deviasi     : {std_val:.4f}")
    print(f"    Hasil           : {'✅ Wajar' if is_sane else f'⚠️  TIDAK WAJAR — {reason}'}")

    return {
        "is_sane"      : is_sane,
        "reason"       : reason,
        "pct_at_ceiling": pct_ceiling,
        "pct_at_floor" : pct_floor,
        "std_val"      : std_val,
    }


# ─── [BARU] FALLBACK: SEASONAL HISTORICAL AVERAGE ────────────────────────────
def _build_seasonal_fallback(full_series: pd.Series,
                              forecast_index: pd.DatetimeIndex,
                              villa_name: str = "") -> pd.DataFrame:
    """
    Bangun forecast pengganti berbasis rata-rata historis per bulan.

    Ini digunakan sebagai fallback ketika SARIMA menghasilkan forecast
    yang tidak wajar (flat ceiling, flat floor, atau flat lainnya).

    Prinsip:
    - Hitung rata-rata & std okupansi per bulan dari data historis
    - Tambahkan sedikit variasi acak (seed=42) agar tidak terlihat flat
    - CI dihitung dari std historis per bulan
    - Pastikan semua nilai dalam range [0.05, 0.97] (menghindari ekstrem)

    Returns DataFrame dengan kolom:
      predicted_occupancy, lower_bound, upper_bound, fallback (=True)
    """
    print(f"\n  [Fallback — {villa_name}] Membangun seasonal historical fallback...")

    # Kembalikan ke skala % untuk perhitungan (lebih intuitif)
    hist = full_series.copy()
    if hist.max() <= 1.5:
        hist = hist * 100.0

    # Rata-rata dan std per bulan dari data historis
    hist_monthly_mean = hist.groupby(hist.index.month).mean()
    hist_monthly_std  = hist.groupby(hist.index.month).std().fillna(5.0)
    overall_mean      = float(hist.mean())
    overall_std       = float(hist.std()) if hist.std() > 0 else 10.0

    print(f"    Historis: mean={overall_mean:.1f}%, std={overall_std:.1f}%")
    print(f"    Per bulan:")
    for m in sorted(hist_monthly_mean.index):
        print(f"      Bulan {m:2d}: mean={hist_monthly_mean[m]:.1f}%  "
              f"std={hist_monthly_std.get(m, overall_std):.1f}%")

    # Bangun prediksi per tanggal di forecast_index
    np.random.seed(42)
    predicted = []
    lower     = []
    upper     = []

    for date in forecast_index:
        m        = date.month
        base     = hist_monthly_mean.get(m, overall_mean)
        std_m    = hist_monthly_std.get(m, overall_std)

        # Sedikit variasi acak agar tidak terlihat flat (30% dari std)
        noise    = np.random.normal(0, std_m * 0.30)
        val_pct  = float(np.clip(base + noise, 5.0, 97.0))

        # CI berbasis std historis bulan tersebut
        ci_half  = std_m * 1.0   # ±1 std historis
        lo_pct   = float(np.clip(val_pct - ci_half, 0.0, 100.0))
        hi_pct   = float(np.clip(val_pct + ci_half, 0.0, 100.0))

        # Simpan dalam skala 0-1 (konsisten dengan SARIMA output)
        predicted.append(val_pct / 100.0)
        lower.append(lo_pct     / 100.0)
        upper.append(hi_pct     / 100.0)

    fallback_df = pd.DataFrame({
        "predicted_occupancy": predicted,
        "lower_bound"        : lower,
        "upper_bound"        : upper,
        "fallback"           : True,
    }, index=forecast_index)

    print(f"    Fallback selesai: mean prediksi = "
          f"{np.mean(predicted)*100:.1f}%  "
          f"std = {np.std(predicted)*100:.1f}%")

    return fallback_df


# ─── DETEKSI m DARI ACF ───────────────────────────────────────────────────────
def _detect_m_from_acf(train: pd.Series, villa_name: str = "") -> int:
    """
    Deteksi seasonal period dominan dari ACF secara otomatis.

    Logika:
    - Cek kandidat lag: 4, 12, 26, 52
    - Hitung ACF di tiap kandidat lag
    - Ambil lag dengan ACF tertinggi yang signifikan (> threshold Bartlett)
    - Kalau tidak ada yang signifikan, fallback ke 4
    """
    candidates = [4, 12, 26, 52]
    n          = len(train)

    max_lag  = min(max(candidates) + 1, n // 2)
    acf_vals = acf(train, nlags=max_lag, fft=True)

    threshold = 1.96 / np.sqrt(n)

    best_m   = 4
    best_val = 0.0

    print(f"  ACF significance (threshold={threshold:.4f}, n={n}):")
    for m in candidates:
        if m < len(acf_vals):
            val = abs(acf_vals[m])
            sig = val > threshold
            print(f"    lag {m:2d}: ACF={val:.4f} "
                  f"{'✅ signifikan' if sig else '❌'}")
            if sig and val > best_val:
                best_val = val
                best_m   = m

    print(f"  → m dari ACF: {best_m} (ACF={best_val:.4f})")
    return best_m


# ─── DETEKSI m SMART ─────────────────────────────────────────────────────────
def _detect_m_smart(train: pd.Series,
                    n_train: int,
                    villa_name: str = "",
                    max_m: int = 52) -> tuple:
    """
    Deteksi m secara cerdas: rule-of-thumb (dari panjang data) + validasi ACF.

    Prinsip utama:
    - Butuh MINIMAL 2 siklus penuh agar SARIMA bisa estimasi seasonal dengan stabil
    - Idealnya 3 siklus (contoh: m=52 butuh 156 minggu = 3 tahun data)
    - Kalau data kurang dari 2 siklus untuk m tertentu → turunkan m
    - ACF memvalidasi apakah pola seasonal memang ada di data atau tidak

    Returns: (m_used, D_used, n_cycles)
    """
    print(f"\n  Deteksi m otomatis [{villa_name}]:")
    print(f"  n_train={n_train} minggu "
          f"= {n_train/52:.1f} tahun = {n_train/26:.1f} semester "
          f"= {n_train/4:.1f} kuartal")

    candidates = [m for m in [4, 12, 26, 52] if m <= max_m]

    print(f"\n  Ketersediaan siklus historis:")
    cycle_info = {}
    for m in candidates:
        n_cyc  = n_train / m
        cycle_info[m] = n_cyc
        status = "✅ ideal" if n_cyc >= 3 else "⚠️  minimum" if n_cyc >= 2 else "❌ kurang"
        print(f"    m={m:2d}: {n_cyc:.1f} siklus {status}")

    eligible = [m for m in candidates if cycle_info[m] >= 2.0]
    if not eligible:
        m_rule = 4
        print(f"\n  ⚠️  Tidak ada m dengan >= 2 siklus, fallback m=4")
    else:
        m_rule = max(eligible)
        print(f"\n  Rule-of-thumb → m={m_rule} "
              f"({cycle_info[m_rule]:.1f} siklus)")

    m_acf = _detect_m_from_acf(train, villa_name)

    print(f"\n  Rekonsiliasi: rule={m_rule}, ACF={m_acf}")

    if m_acf == m_rule:
        m_final = m_rule
        print(f"  ✅ Sepakat → m={m_final}")
    elif m_acf > m_rule:
        m_final = m_rule
        print(f"  ⚠️  ACF suggest m={m_acf} tapi data hanya cukup "
              f"untuk m={m_rule} → pakai m={m_final}")
    else:
        if m_acf in cycle_info and cycle_info[m_acf] >= 2:
            m_final = m_acf
            print(f"  ℹ️  ACF={m_acf} < rule={m_rule}, "
                  f"pola m={m_rule} tidak terkonfirmasi → m={m_final}")
        else:
            m_final = m_rule
            print(f"  ℹ️  ACF={m_acf} tapi siklus tidak cukup → m={m_final}")

    n_cycles_final = cycle_info.get(m_final, n_train / m_final)
    D_used         = 1 if m_final > 1 else 0

    print(f"\n  ✅ m FINAL = {m_final} "
          f"({n_cycles_final:.1f} siklus historis, D={D_used})")

    return m_final, D_used, n_cycles_final


# ─── FIT SARIMA ───────────────────────────────────────────────────────────────
def fit_sarima(series: pd.Series,
               villa_name: str,
               seasonal_period: int = None,
               test_size: float = None) -> dict:

    if not STATSMODELS_OK:
        return None

    max_m    = seasonal_period if seasonal_period is not None else CONFIG["seasonal_period"]
    if test_size is None:
        test_size = CONFIG["test_size"]

    # ── 1. Resample ke mingguan ───────────────────────────────────
    s = series.resample('W').mean().dropna()

    # ── 2. Normalize ke 0-1 ──────────────────────────────────────
    s_max = s.max()
    if s_max > 1.5:
        s = s / 100.0
    s = s.clip(0, 1)

    n = len(s)

    # ── 3. Split train/test ───────────────────────────────────────
    n_test  = max(int(n * test_size), 4)
    n_train = n - n_test
    train   = s.iloc[:n_train]
    test    = s.iloc[n_train:]

    print(f"\n{'='*60}")
    print(f"[{villa_name}] SARIMA Fitting")
    print(f"  Data range : {s.index[0].date()} → {s.index[-1].date()} (n={n})")
    print(f"  Train      : {train.index[0].date()} → {train.index[-1].date()} "
          f"(n={n_train})")
    print(f"  Test       : {test.index[0].date()} → {test.index[-1].date()} "
          f"(n={n_test})")
    print(f"  Skala data : {s.min():.4f} – {s.max():.4f}")

    # ── [BARU] 3b. Ceiling data check ────────────────────────────
    # Deteksi sejak dini apakah data didominasi ceiling/floor.
    # Ini tidak menghentikan proses training, tapi hasilnya
    # disimpan di metadata agar bisa jadi sinyal di UI dan
    # digunakan oleh sanity check saat forecast.
    ceiling_info = _detect_ceiling_data(s, villa_name)

    # ── 4. Deteksi m otomatis ─────────────────────────────────────
    m_used, D_used, n_cycles = _detect_m_smart(
        train, n_train, villa_name, max_m=max_m
    )

    # ── 5. Validasi minimum data ──────────────────────────────────
    min_train_needed = max(2 * m_used + 4, 12)
    if n_train < min_train_needed:
        print(f"  ⚠️  Data terlalu pendek: n_train={n_train}, "
              f"butuh minimal {min_train_needed} minggu (m={m_used}), skip.")
        return None

    # ── 6. ADF pre-test untuk d ───────────────────────────────────
    d_fixed = 1
    try:
        adf_pval = adfuller(train, autolag='AIC')[1]
        d_fixed  = 0 if adf_pval < 0.05 else 1
        print(f"\n  ADF p-value: {adf_pval:.4f} → d={d_fixed} "
              f"(stasioner={adf_pval < 0.05})")
    except Exception as adf_err:
        print(f"\n  ADF gagal ({adf_err}), fallback d=1")

    # ── 7. auto_arima ─────────────────────────────────────────────
    print('  Mencari parameter SARIMA...')

    if not PMDARIMA_OK:
        print('  ⚠️  pmdarima tidak tersedia, gunakan order default.')
        order   = (1, 1, 1)
        s_order = (1, 1, 1, m_used)
    else:
        auto_model = auto_arima(
            train,
            start_p=0, max_p=2,
            start_q=0, max_q=2,
            d=d_fixed, max_d=1,
            start_P=0, max_P=1,
            start_Q=0, max_Q=1,
            D=D_used,
            m=m_used,
            seasonal=True,
            information_criterion='aic',
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            random_state=42,
            n_jobs=1,
            max_order=5,
            with_ols=False,
        )
        order   = auto_model.order
        s_order = auto_model.seasonal_order

    print(f"  ARIMA order    : {order}")
    print(f"  Seasonal order : {s_order}")
    print(f"  m digunakan    : {m_used} ({n_cycles:.1f} siklus dari data)")

    # ── 8. Refit dengan SARIMAX ───────────────────────────────────
    results = SARIMAX(
        train,
        order=order,
        seasonal_order=s_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
        hamilton_representation=False,
    ).fit(
        disp=False,
        method='lbfgs',
        maxiter=200,
        cov_type='none',
        low_memory=True,
    )

    aic = results.aic
    print(f"  AIC (normalized scale) : {aic:.2f}")

    # ── [BARU] AIC display: skala asli (persen, bukan 0-1) ───────
    # AIC dihitung dari data ternormalisasi (0-1), sehingga nilainya
    # jauh lebih kecil dari referensi literatur yang pakai skala %.
    # Kita simpan versi "display" agar UI tidak membingungkan.
    # Rumus aproksimasi: AIC_pct ≈ AIC_norm + 2 * n_train * log(100)
    # karena log-likelihood berubah sebesar n*log(scale_factor)
    aic_display = aic + 2 * n_train * np.log(100)
    print(f"  AIC (display, ~pct scale): {aic_display:.0f}")

    # ── 9. Evaluasi test set ──────────────────────────────────────
    forecast_vals = results.forecast(steps=len(test)).clip(0, 1)
    y_true        = test.values
    y_pred        = forecast_vals.values

    # Metrics dalam skala PERSEN (lebih intuitif di UI)
    mae_pct  = _mae(y_true * 100, y_pred * 100)
    rmse_pct = _rmse(y_true * 100, y_pred * 100)
    mape_val = _mape(y_true, y_pred)   # MAPE tidak terpengaruh skala

    print(f"\n  📊 Test Set (skala %): "
          f"MAE={mae_pct:.2f}pp | RMSE={rmse_pct:.2f}pp | MAPE={mape_val:.2f}%")

    return {
        'villa'              : villa_name,
        'model'              : results,
        'train'              : train,
        'test'               : test,
        'forecast'           : forecast_vals,
        'order'              : order,
        'seasonal_order'     : s_order,
        'm_used'             : m_used,
        'n_cycles'           : round(n_cycles, 2),
        'aic'                : aic_display,   # ← display version (approx pct scale)
        'aic_raw'            : aic,           # ← raw (normalized scale, untuk internal)
        'mae'                : mae_pct,
        'rmse'               : rmse_pct,
        'mape'               : mape_val,
        'series'             : s,
        'rating'             : _rating(mape_val),
        'ceiling_info'       : ceiling_info,  # ← [BARU] info ceiling/floor detection
    }


# ─── DB: SAVE MODEL ───────────────────────────────────────────────────────────
def save_model_to_db(villa_name: str, model, meta_data: dict, series) -> bool:
    """Simpan model + meta ke TiDB sebagai BLOB."""
    model_buffer = io.BytesIO()
    meta_buffer  = io.BytesIO()
    joblib.dump(model, model_buffer, compress=("zlib", 6))
    joblib.dump({**meta_data, "_series": series}, meta_buffer, compress=3)

    model_bytes = model_buffer.getvalue()
    meta_bytes  = meta_buffer.getvalue()

    conn = get_conn()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sarima_models
                (villa_name, model_blob, meta_blob, mape, rmse, aic)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_blob = VALUES(model_blob),
                meta_blob  = VALUES(meta_blob),
                mape       = VALUES(mape),
                rmse       = VALUES(rmse),
                aic        = VALUES(aic),
                trained_at = NOW()
        """, (
            villa_name,
            model_bytes,
            meta_bytes,
            meta_data.get("mape"),
            meta_data.get("rmse"),
            meta_data.get("aic"),   # aic_display yang disimpan ke DB
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ save_model_to_db error: {e}")
        return False


# ─── DB: LOAD MODEL ───────────────────────────────────────────────────────────
def load_model_from_db(villa_name: str) -> tuple:
    """Load model + meta dari TiDB. Returns (model, meta_data) atau (None, None)."""
    conn = get_conn()
    if not conn:
        return None, None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT model_blob, meta_blob FROM sarima_models WHERE villa_name=%s",
            (villa_name,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            return None, None

        model     = joblib.load(io.BytesIO(row["model_blob"]))
        meta_data = joblib.load(io.BytesIO(row["meta_blob"]))
        return model, meta_data
    except Exception as e:
        print(f"❌ load_model_from_db error: {e}")
        return None, None


# ─── DB: CHECK MODEL EXISTS ───────────────────────────────────────────────────
def model_exists_db(villa_name: str) -> bool:
    result = run_query(
        "SELECT id FROM sarima_models WHERE villa_name=%s",
        (villa_name,)
    )
    return result is not None and not result.empty


# ─── CACHE INVALIDATION ───────────────────────────────────────────────────────
def _invalidate_model_cache(villa_name: str = None):
    if villa_name:
        _model_exists_cache.pop(villa_name, None)
    else:
        _model_exists_cache.clear()


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

        sub         = df_occ[df_occ["villa_name"] == villa_name].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        series      = sub.set_index("date")["occupancy_pct"].sort_index()

        window = CONFIG.get("rolling_window_days")
        if window and len(series) > window:
            series = series.iloc[-window:]

        res = fit_sarima(series, villa_name)

        if res is not None:
            joblib.dump(res["model"], pkl, compress=("zlib", 6))
            meta_data = {k: v for k, v in res.items()
                         if k not in ("model", "train", "test", "forecast", "series")}
            meta_data["status"]      = "trained"
            meta_data["data_end"]    = str(res["series"].index[-1].date())
            meta_data["n_train"]     = len(res["train"])
            meta_data["n_test"]      = len(res["test"])
            meta_data["model_path"]  = pkl
            meta_data["_series_end"] = res["series"].index[-1]
            joblib.dump({**meta_data, "_series": res["series"]}, meta, compress=3)

            save_model_to_db(villa_name, res["model"], meta_data, res["series"])
            _invalidate_model_cache(villa_name)

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

    if not force_retrain:
        if os.path.exists(pkl) and os.path.exists(meta):
            cached = joblib.load(meta)
            cached["status"] = "loaded_from_cache"
            return cached

        model, meta_data = load_model_from_db(villa_name)
        if model is not None and meta_data is not None:
            joblib.dump(model, pkl, compress=("zlib", 6))
            joblib.dump(meta_data, meta, compress=3)
            meta_data["status"] = "loaded_from_db"
            return meta_data

    window = CONFIG.get("rolling_window_days")
    if window and len(series) > window:
        series = series.iloc[-window:]

    try:
        res = fit_sarima(series, villa_name)
    except Exception as e:
        import traceback
        return {
            "status"   : "error",
            "message"  : f"fit_sarima exception: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }

    if res is None:
        s_check = series.resample('W').mean().dropna()
        if s_check.max() > 1.5:
            s_check = s_check / 100.0
        n       = len(s_check)
        n_test  = max(int(n * CONFIG["test_size"]), 4)
        n_train = n - n_test
        if not STATSMODELS_OK:
            msg = "statsmodels tidak terinstall"
        else:
            msg = (f"Data terlalu pendek atau pola seasonal tidak terdeteksi: "
                   f"{n} minggu total, n_train={n_train}.")
        return {"status": "error", "message": msg}

    joblib.dump(res["model"], pkl, compress=("zlib", 6))
    meta_data = {k: v for k, v in res.items()
                 if k not in ("model", "train", "test", "forecast")}
    meta_data["status"]     = "trained"
    meta_data["data_end"]   = str(res["series"].index[-1].date())
    meta_data["n_train"]    = len(res["train"])
    meta_data["n_test"]     = len(res["test"])
    meta_data["model_path"] = pkl
    joblib.dump({**meta_data, "_series": res["series"]}, meta, compress=3)

    save_model_to_db(villa_name, res["model"], meta_data, res["series"])
    _invalidate_model_cache(villa_name)

    return meta_data


# ─── FORECAST ─────────────────────────────────────────────────────────────────
def forecast(villa_name: str,
             horizon: int = 26,
             target_end_date: str = "2026-06-30") -> pd.DataFrame:
    """
    Generate forecast untuk villa_name hingga target_end_date.

    Alur lengkap:
    1. Load model dari disk / DB
    2. Hitung steps yang dibutuhkan
    3. Generate forecast SARIMA
    4. Filter ke range tanggal yang diminta
    5. [BARU] Sanity check — apakah hasil forecast wajar?
    6. [BARU] Kalau tidak wajar → fallback ke seasonal historical average
    7. Return DataFrame dengan kolom standar + kolom 'fallback' (bool)
    """
    pkl  = _pkl_path(villa_name)
    meta = _meta_path(villa_name)

    if not os.path.exists(pkl):
        model, meta_data = load_model_from_db(villa_name)
        if model is None:
            return pd.DataFrame({"error": ["Model belum ditraining."]})
        joblib.dump(model, pkl, compress=("zlib", 6))
        joblib.dump(meta_data, meta, compress=3)

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

        # ── Generate SARIMA forecast ──────────────────────────────
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

        # ── Filter ke range tahun target ─────────────────────────
        target_year  = target_date.year
        target_month = target_date.month
        mask = (
            (forecast_df.index.year == target_year) &
            (forecast_df.index.month >= 1) &
            (forecast_df.index.month <= target_month)
        )
        forecast_2026 = forecast_df[mask].copy()

        if len(forecast_2026) == 0:
            return pd.DataFrame({"error": [
                "Tidak ada data untuk range tanggal yang diminta."
            ]})

        # ── [BARU] Sanity check forecast ──────────────────────────
        # Periksa apakah hasil SARIMA wajar sebelum dikembalikan ke UI.
        # Kalau tidak wajar, ganti dengan fallback berbasis historis.
        sanity = _sanity_check_forecast(
            forecast_2026["predicted_occupancy"], villa_name
        )

        if not sanity["is_sane"]:
            print(f"\n  ⚠️  [{villa_name}] Forecast tidak wajar: {sanity['reason']}")
            print(f"       Beralih ke seasonal historical fallback...")

            fallback_df = _build_seasonal_fallback(
                full_series,
                forecast_2026.index,
                villa_name,
            )
            # Ganti kolom forecast dengan fallback, pertahankan index
            forecast_2026["predicted_occupancy"] = fallback_df["predicted_occupancy"].values
            forecast_2026["lower_bound"]         = fallback_df["lower_bound"].values
            forecast_2026["upper_bound"]         = fallback_df["upper_bound"].values
            forecast_2026["fallback"]            = True
            forecast_2026["fallback_reason"]     = sanity["reason"]
        else:
            forecast_2026["fallback"]        = False
            forecast_2026["fallback_reason"] = ""

        # ── Tambah kolom label ────────────────────────────────────
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
    print(f"  Data terakhir  : {last_date.date()}")
    print(f"  Weeks predicted: {weeks_to_predict}")
    print(f"  m digunakan    : {sarima_result.get('m_used','?')} "
          f"({sarima_result.get('n_cycles','?')} siklus historis)")

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

    # ── [BARU] Sanity check di predict_2026 juga ─────────────────
    sanity = _sanity_check_forecast(
        forecast_2026["predicted_occupancy"], villa_name
    )
    if not sanity["is_sane"]:
        print(f"  ⚠️  Forecast tidak wajar di predict_2026: {sanity['reason']}")
        fallback_df = _build_seasonal_fallback(
            full_series, forecast_2026.index, villa_name
        )
        forecast_2026["predicted_occupancy"] = fallback_df["predicted_occupancy"].values
        forecast_2026["lower_bound"]         = fallback_df["lower_bound"].values
        forecast_2026["upper_bound"]         = fallback_df["upper_bound"].values
        forecast_2026["fallback"]            = True
    else:
        forecast_2026["fallback"] = False

    forecast_2026_copy               = forecast_2026.copy()
    forecast_2026_copy['month']      = forecast_2026_copy.index.month
    forecast_2026_copy['month_name'] = forecast_2026_copy.index.strftime('%B')
    monthly_avg = forecast_2026_copy.groupby(
        ['month', 'month_name']
    )['predicted_occupancy'].mean()

    print(f"📊 Rata-rata Jan-Jun 2026: "
          f"{forecast_2026['predicted_occupancy'].mean()*100:.2f}%")
    for (mn, mname), avg in monthly_avg.items():
        flag = " [FALLBACK]" if forecast_2026["fallback"].any() else ""
        print(f"   {mname:10s}: {avg*100:6.2f}%{flag}")

    return {
        'villa'        : villa_name,
        'forecast_full': forecast_df,
        'forecast_2026': forecast_2026,
        'monthly_avg'  : monthly_avg,
        'full_series'  : full_series,
        'model'        : model,
        'fallback'     : bool(forecast_2026["fallback"].any()),
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
            # Ambil ceiling_info kalau ada
            ceiling_info = meta.get("ceiling_info", {})
            rows.append({
                "Vila"             : name,
                "Order"            : str(meta.get("order", "-")),
                "Seasonal Order"   : str(meta.get("seasonal_order", "-")),
                "m digunakan"      : meta.get("m_used", "-"),
                "Siklus"           : meta.get("n_cycles", "-"),
                "AIC"              : meta.get("aic", "-"),       # display version
                "MAE"              : meta.get("mae", "-"),
                "RMSE"             : meta.get("rmse", "-"),
                "MAPE (%)"         : meta.get("mape", "-"),
                "Rating"           : meta.get("rating", "-"),
                "Status"           : meta.get("status", "-"),
                "Data Hingga"      : meta.get("data_end", "-"),
                "Ceiling Dominated": ceiling_info.get("is_ceiling_dominated", "-"),
                "% Ceiling"        : ceiling_info.get("pct_at_ceiling", "-"),
            })
        else:
            rows.append({
                "Vila"             : name,
                "Order"            : "-",
                "Seasonal Order"   : "-",
                "m digunakan"      : "-",
                "Siklus"           : "-",
                "AIC"              : "-",
                "MAE"              : "-",
                "RMSE"             : "-",
                "MAPE (%)"         : "-",
                "Rating"           : "⏳ Belum ditraining",
                "Status"           : "not_trained",
                "Data Hingga"      : "-",
                "Ceiling Dominated": "-",
                "% Ceiling"        : "-",
            })
    return pd.DataFrame(rows)


# ─── MODEL EXISTS ─────────────────────────────────────────────────────────────
def model_exists(villa_name: str) -> bool:
    if villa_name in _model_exists_cache:
        return _model_exists_cache[villa_name]
    if os.path.exists(_pkl_path(villa_name)):
        _model_exists_cache[villa_name] = True
        return True
    result = model_exists_db(villa_name)
    _model_exists_cache[villa_name] = result
    return result


def models_status(villa_names: list) -> dict:
    return {
        name: "trained" if model_exists(name) else "not_trained"
        for name in villa_names
    }


def engine_info() -> dict:
    return {
        "statsmodels_ok"   : STATSMODELS_OK,
        "pmdarima_ok"      : PMDARIMA_OK,
        "data_frequency"   : "weekly",
        "seasonal_period"  : f"auto (max={CONFIG['seasonal_period']})",
        "m_detection"      : "ACF significance + rule-of-thumb (min 2 cycles)",
        "m_candidates"     : [4, 12, 26, 52],
        "random_state"     : 42,
        "max_p_q"          : 2,
        "max_P_Q"          : 1,
        "optimizer"        : "lbfgs",
        "max_order"        : 5,
        "cov_type"         : "none",
        "sanity_check"     : "enabled — fallback to seasonal historical average",
        "ceiling_detection": "enabled — deteksi data ceiling/floor dominated",
        "aic_display"      : "approx pct-scale (aic_raw + 2*n_train*log(100))",
        "config"           : CONFIG,
    }