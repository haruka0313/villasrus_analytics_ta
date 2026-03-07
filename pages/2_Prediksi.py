import streamlit as st

# ─── PAGE CONFIG — HARUS PALING ATAS SEBELUM APAPUN ──────────────────────────
st.set_page_config(
    page_title="Dashboard — Villas R Us",
    page_icon="🏝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.cache = st.cache_data
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from utils.auth import get_cookie_manager, set_session, load_from_cookie
from utils.sidebar import render_sidebar

cookies = get_cookie_manager()

from utils.page_guard import require_login
require_login(cookies)

render_sidebar(cookies)

from database import (
    get_occupancy_data, get_financial_data, get_villas,
    get_all_sarima_models,
)

# ─── SARIMA IMPORTS ───────────────────────────────────────────────────────────
SARIMA_OK         = False
TRAIN_OK          = False
SARIMA_ERR        = ""
sarima_forecast   = None
sarima_exists     = None
sarima_train      = None
sarima_load_fc_db = None

try:
    from utils.sarima_engine import forecast as sarima_forecast
    from utils.sarima_engine import model_exists as sarima_exists
    from utils.sarima_engine import load_forecast_from_db as sarima_load_fc_db
    SARIMA_OK = True
except Exception as e:
    SARIMA_ERR = str(e)

try:
    from utils.sarima_engine import train_and_save as sarima_train
    TRAIN_OK = True
except Exception:
    pass

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background: #f0f6ff !important; color: #0f172a; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stSidebarNav"] { display: none !important; }
  [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
  section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; color: #334155 !important;
    border: 1px solid #e2e8f0 !important; box-shadow: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f0f6ff !important; border-color: #38bdf8 !important; color: #0369a1 !important;
  }
  .page-eyebrow { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: .18em; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
  .page-title { font-size: 22px; font-weight: 800; color: #0f172a; margin-bottom: 4px; }
  .page-title em { font-style: italic; color: #0369a1; }
  .page-sub { font-size: 13px; color: #94a3b8; }
  .rule { border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }
  .section-label {
    font-size: 11px; font-weight: 700; color: #94a3b8;
    letter-spacing: .12em; text-transform: uppercase;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0; font-family: 'Sora', sans-serif;
  }
  .model-badge { display: inline-flex; align-items: center; gap: 6px; font-family: 'DM Mono', monospace; font-size: 10px; padding: 4px 12px; border-radius: 100px; }
  .model-badge.ok   { background: #dcfce7; border: 1px solid #bbf7d0; color: #166534; }
  .model-badge.warn { background: #fef9c3; border: 1px solid #fde68a; color: #854d0e; }
  .model-badge.err  { background: #fee2e2; border: 1px solid #fecaca; color: #991b1b; }
  .exec-metric {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(3,105,161,0.06);
  }
  .exec-metric-label { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
  .exec-metric-value { font-size: 28px; font-weight: 800; color: #0f172a; font-family: 'DM Mono', monospace; margin-bottom: 4px; }
  .exec-metric-delta { font-size: 12px; color: #64748b; }
  .exec-metric-delta.positive { color: #166534; }
  .exec-metric-delta.negative { color: #991b1b; }
  .forecast-table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; font-size: 13px; }
  .forecast-table th { background: #f8fafc; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: #94a3b8; padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }
  .forecast-table td { padding: 12px 16px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  .forecast-table tr:last-child td { border-bottom: none; }
  .forecast-table tr:hover td { background: #f8fafc; }
  .forecast-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
  .forecast-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #f1f5f9; }
  .forecast-table-wrapper { display: block; }
  .forecast-cards-wrapper { display: none; }
  @media (max-width: 768px) {
    .forecast-table-wrapper { display: none; }
    .forecast-cards-wrapper { display: block; }
  }
  .summary-acc-table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; font-size: 13px; }
  .summary-acc-table th { background: #0f172a; font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: #94a3b8; padding: 12px 16px; text-align: left; border-bottom: 1px solid #1e293b; }
  .summary-acc-table td { padding: 11px 16px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  .summary-acc-table tr:last-child td { border-bottom: none; }
  .summary-acc-table tr:hover td { background: #f8fafc; }
  .summary-acc-table .villa-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
  .stButton > button { border-radius: 8px !important; font-weight: 600 !important; font-size: 13px !important; }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
  .anim { animation: fadeUp .4s ease both; }
  .cache-badge { display: inline-flex; align-items: center; gap: 5px; font-family: 'DM Mono', monospace; font-size: 9px; padding: 3px 10px; border-radius: 100px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# ─── VILLA INSIGHTS ───────────────────────────────────────────────────────────
VILLA_INSIGHTS = {
    "Briana Villas":  {"elastisitas": "Semi-Elastis", "color": "#3D6BE8", "area": "Canggu",   "price_floor": 1.2, "price_ceiling": 3.5, "naik_pct": 15, "turun_pct": 10, "elasticity_factor": 0.15, "strategi_peak": "Naikkan rate saat occupancy >85%. Tutup diskon.",              "strategi_low": "Tawarkan paket 3-malam dengan diskon 10%. Early bird promo."},
    "Castello Villas":{"elastisitas": "Semi-Elastis", "color": "#7c3aed", "area": "Canggu",   "price_floor": 1.0, "price_ceiling": 3.0, "naik_pct": 12, "turun_pct": 12, "elasticity_factor": 0.15, "strategi_peak": "Konsistensi pricing. Batasi OTA commission di peak.",           "strategi_low": "Flash sale 12% + paket F&B bundling."},
    "Elina Villas":   {"elastisitas": "Semi-Elastis", "color": "#059669", "area": "Canggu",   "price_floor": 3.0, "price_ceiling": 3.5, "naik_pct": 8,  "turun_pct": 5,  "elasticity_factor": 0.12, "strategi_peak": "Pertahankan harga premium Rp3-3.5M. Prioritas direct booking.", "strategi_low": "Fokus high season saja. Tutup operasional off-season."},
    "Isola Villas":   {"elastisitas": "Elastis",      "color": "#db2777", "area": "Canggu",   "price_floor": 1.5, "price_ceiling": 4.0, "naik_pct": 20, "turun_pct": 15, "elasticity_factor": 0.25, "strategi_peak": "Naikkan agresif +20%. Weekend packages. Minimum stay 2 malam.", "strategi_low": "Flash sale 15%. Paket mid-week spesial."},
    "Eindra Villas":  {"elastisitas": "Inelastis",    "color": "#d97706", "area": "Seminyak", "price_floor": 4.0, "price_ceiling": 7.7, "naik_pct": 5,  "turun_pct": 3,  "elasticity_factor": 0.05, "strategi_peak": "Pertahankan Rp4-5M. Potensi naik tipis +5%.",                  "strategi_low": "Tawarkan kontrak korporat & long-stay. Jangan turun drastis."},
    "Esha Villas":    {"elastisitas": "Semi-Elastis", "color": "#b45309", "area": "Seminyak", "price_floor": 1.3, "price_ceiling": 4.3, "naik_pct": 18, "turun_pct": 12, "elasticity_factor": 0.18, "strategi_peak": "Terapkan Rp3-4M di Mei-Oktober. Rate optimizer aktif.",           "strategi_low": "Turunkan ke Rp1.5-2M Nov-Apr. Paket liburan sekolah."},
    "Ozamiz Villas":  {"elastisitas": "Inelastis",    "color": "#9333ea", "area": "Seminyak", "price_floor": 2.0, "price_ceiling": 5.0, "naik_pct": 4,  "turun_pct": 2,  "elasticity_factor": 0.04, "strategi_peak": "Fokus long-stay & kontrak korporat.",                           "strategi_low": "Pertahankan harga kontrak. Jangan diskon."},
}

LAYOUT      = dict(plot_bgcolor="#ffffff", paper_bgcolor="#F7F7F5", font_color="#555", font_family="DM Sans")
PEAK_MONTHS = [6, 7, 8, 12, 1]


def safe_hex(hex_color):
    h = hex_color.lstrip("#")
    if len(h) != 6: h = "3D6BE8"
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    return get_occupancy_data(), get_financial_data(), get_villas()

@st.cache_data(ttl=300)
def load_sarima_summary():
    """Metrik semua model dari sarima_models table."""
    try:
        df = get_all_sarima_models()
        return df if (df is not None and not df.empty) else pd.DataFrame()
    except Exception as e:
        st.warning(f"Gagal load SARIMA summary: {e}")
        return pd.DataFrame()

def get_sarima_db_row(villa_name: str, summary_df: pd.DataFrame) -> dict:
    try:
        if summary_df is None or summary_df.empty: return {}
        match = summary_df[summary_df["villa_name"] == villa_name]
        return match.iloc[0].to_dict() if not match.empty else {}
    except Exception:
        return {}


df_occ, df_fin, df_villas = load_data()
sarima_summary_db           = load_sarima_summary()

if df_occ is None or df_occ.empty:
    st.warning("Belum ada data. Silakan upload terlebih dahulu.")
    if st.button("Upload Data"): st.switch_page("pages/3_Upload.py")
    st.stop()

df_occ = df_occ.copy()
df_occ["date"]      = pd.to_datetime(df_occ["date"])
df_occ["month"]     = df_occ["date"].dt.to_period("M").astype(str)
df_occ["month_num"] = df_occ["date"].dt.month
df_occ["year"]      = df_occ["date"].dt.year

if df_fin is not None and not df_fin.empty:
    df_fin = df_fin.copy()
    df_fin["date"]      = pd.to_datetime(df_fin["date"])
    df_fin["month"]     = df_fin["date"].dt.to_period("M").astype(str)
    df_fin["month_num"] = df_fin["date"].dt.month
    df_fin["year"]      = df_fin["date"].dt.year

villa_names  = sorted(df_occ["villa_name"].unique().tolist())
VILLA_COLORS = {}
if df_villas is not None and not df_villas.empty:
    VILLA_COLORS = {r["villa_name"]: r["color_hex"] for _, r in df_villas.iterrows()}

def get_color(n):
    return VILLA_COLORS.get(n, VILLA_INSIGHTS.get(n, {}).get("color", "#3D6BE8"))


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='anim'>
  <div class='page-eyebrow'>Integrated Analysis · SARIMA Forecast</div>
  <div class='page-title'>Strategi <em>Okupansi 2026</em></div>
  <div class='page-sub'>Prediksi okupansi dengan analisis komponen SARIMA dan pola historis</div>
</div>
<hr class='rule'>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>📊 Executive Summary 2026</div>", unsafe_allow_html=True)

hist_occ_all       = df_occ["occupancy_pct"].mean() if not df_occ.empty else 70.0
peak_mask_all      = df_occ["month_num"].isin(PEAK_MONTHS)
peak_avg_all       = df_occ.loc[peak_mask_all, "occupancy_pct"].mean() if peak_mask_all.any() else 0.0
risk_villas_count  = 0
best_villa_overall = "—"
best_occ_overall   = 0.0

for vn in villa_names:
    vd_tmp  = df_occ[df_occ["villa_name"] == vn]
    avg_tmp = vd_tmp["occupancy_pct"].mean() if not vd_tmp.empty else 0
    if avg_tmp < 50: risk_villas_count += 1
    if avg_tmp > best_occ_overall:
        best_occ_overall   = avg_tmp
        best_villa_overall = vn

occ_2026_actual = df_occ[df_occ["year"] == 2026]["occupancy_pct"].mean() \
                  if (df_occ["year"] == 2026).any() else None

e1, e2, e3, e4 = st.columns(4)
with e1:
    val_disp = f"{occ_2026_actual:.1f}%" if occ_2026_actual is not None else f"{hist_occ_all:.1f}%"
    lbl_disp = "Avg Okupansi 2026 (Aktual)" if occ_2026_actual is not None else "Avg Okupansi Historis"
    st.markdown(f"""<div class='exec-metric'><div class='exec-metric-label'>{lbl_disp}</div>
        <div class='exec-metric-value'>{val_disp}</div><div class='exec-metric-delta'>Semua vila</div></div>""", unsafe_allow_html=True)
with e2:
    st.markdown(f"""<div class='exec-metric'><div class='exec-metric-label'>Peak Season Average</div>
        <div class='exec-metric-value'>{peak_avg_all:.1f}%</div><div class='exec-metric-delta'>Jun–Aug, Des–Jan</div></div>""", unsafe_allow_html=True)
with e3:
    rc2 = "negative" if risk_villas_count > 3 else "positive" if risk_villas_count == 0 else ""
    st.markdown(f"""<div class='exec-metric'><div class='exec-metric-label'>Vila Rata-rata &lt;50%</div>
        <div class='exec-metric-value'>{risk_villas_count}/{len(villa_names)}</div>
        <div class='exec-metric-delta {rc2}'>ocupansi rata-rata &lt;50%</div></div>""", unsafe_allow_html=True)
with e4:
    st.markdown(f"""<div class='exec-metric'><div class='exec-metric-label'>Vila Terbaik</div>
        <div class='exec-metric-value' style='font-size:18px'>{best_villa_overall.replace(" Villas","")}</div>
        <div class='exec-metric-delta'>highest historical occupancy</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VILLA SELECTOR + MODEL STATUS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='rule'>", unsafe_allow_html=True)

cc1, cc2 = st.columns([2.5, 4])
with cc1:
    sel_villa = st.selectbox("Vila", villa_names, label_visibility="collapsed", key="sv")
    st.markdown("<div style='font-size:11px;color:#aaa;margin-top:-8px'>Pilih vila</div>", unsafe_allow_html=True)
with cc2:
    show_ci = st.toggle("Confidence Interval", value=True, key="sci")

vc            = get_color(sel_villa)
sel_db_row    = get_sarima_db_row(sel_villa, sarima_summary_db)

# ── FIX: Cek model_trained dengan aman ───────────────────────────────────────
model_trained = False
try:
    if SARIMA_OK and sarima_exists is not None and callable(sarima_exists):
        model_trained = bool(sarima_exists(sel_villa))
except Exception as e:
    st.warning(f"Gagal cek status model: {e}")
    model_trained = False

st.markdown("<div class='section-label'>Model SARIMA — Status &amp; Train</div>", unsafe_allow_html=True)

# ── FIX: Seluruh blok t1-t4 dibungkus try/except ─────────────────────────────
try:
    t1, t2, t3, t4 = st.columns([2.5, 1.5, 1.5, 3])

    with t1:
        if not SARIMA_OK:   bcls, btxt = "err",  "Import Error"
        elif model_trained: bcls, btxt = "ok",   "Model Tersedia"
        else:               bcls, btxt = "warn", "Belum Dilatih"

        fc_cached_badge = ""
        if SARIMA_OK and model_trained and sarima_load_fc_db is not None and callable(sarima_load_fc_db):
            try:
                cached_check = sarima_load_fc_db(sel_villa, year=2026)
                if cached_check is not None and not cached_check.empty:
                    gen_at  = cached_check["generated_at"].iloc[0] if "generated_at" in cached_check.columns else ""
                    gen_str = str(gen_at)[:16]
                    fc_cached_badge = f"<span class='cache-badge' style='margin-left:8px'>💾 cached · {gen_str}</span>"
            except Exception:
                pass

        st.markdown(f"""<div style='padding:10px 0'>
          <span class='model-badge {bcls}'>{btxt}</span>
          <span style='font-size:11px;color:#aaa;margin-left:10px;font-family:DM Mono,monospace'>{sel_villa}</span>
          {fc_cached_badge}
        </div>""", unsafe_allow_html=True)

    with t2:
        if TRAIN_OK and sarima_train is not None and not model_trained:
            if st.button("🚀 Train Model", use_container_width=True, key="btn_train"):
                with st.spinner(f"Melatih {sel_villa}... (3-8 menit)"):
                    try:
                        vd     = df_occ[df_occ["villa_name"] == sel_villa].copy()
                        series = vd.set_index("date")["occupancy_pct"].sort_index()
                        result = sarima_train(sel_villa, series)
                        if result.get("status") == "trained":
                            st.success(f"Selesai! MAPE: {result.get('mape','?')}%")
                            st.cache_data.clear(); st.rerun()
                        else:
                            st.error(f"Gagal: {result.get('message','unknown')}")
                            if result.get("traceback"): st.code(result["traceback"], language="python")
                    except Exception as e:
                        import traceback
                        st.error(f"Error: {e}"); st.code(traceback.format_exc(), language="python")
        else:
            st.button("🚀 Train Model", use_container_width=True, disabled=True, key="btn_train_dis")

    with t3:
        if TRAIN_OK and sarima_train is not None and model_trained:
            if st.button("🔄 Retrain", use_container_width=True, key="btn_retrain"):
                with st.spinner(f"Retrain {sel_villa}..."):
                    try:
                        vd     = df_occ[df_occ["villa_name"] == sel_villa].copy().sort_values("date")
                        series = vd.set_index("date")["occupancy_pct"].sort_index()
                        result = sarima_train(sel_villa, series, force_retrain=True)
                        if result.get("status") == "trained":
                            st.success(f"Retrain selesai! MAPE: {result.get('mape','?')}%")
                            st.cache_data.clear(); st.rerun()
                        else:
                            st.error(f"Gagal: {result.get('message','unknown')}")
                            if result.get("traceback"): st.code(result["traceback"], language="python")
                    except Exception as e:
                        import traceback
                        st.error(f"Error: {e}"); st.code(traceback.format_exc(), language="python")
        else:
            st.button("🔄 Retrain", use_container_width=True, disabled=True, key="btn_retrain_dis")

    with t4:
        if not SARIMA_OK:
            st.markdown(f"<div style='font-size:11px;color:#C0392B;background:#FFF0F0;border:1px solid #FFC5C5;border-radius:8px;padding:8px 12px;font-family:DM Mono,monospace'>❌ Import error: {SARIMA_ERR[:120]}</div>", unsafe_allow_html=True)
        elif not model_trained:
            st.markdown("<div style='font-size:11px;color:#A05C00;background:#FFF8EC;border:1px solid #FFD88A;border-radius:8px;padding:8px 12px'>⚠️ Model belum dilatih. Klik <b>Train Model</b> untuk memulai.</div>", unsafe_allow_html=True)
        elif sel_db_row:
            order_str  = f"{sel_db_row.get('arima_order','?')}{sel_db_row.get('seasonal_order','')}"
            n_train    = sel_db_row.get("n_train", "?")
            aic_val    = sel_db_row.get("aic", "?")
            m_val      = sel_db_row.get("m_used", "?")
            mape_val   = sel_db_row.get("mape", "?")
            trained_at = str(sel_db_row.get("trained_at", ""))[:16]
            st.markdown(f"""<div style='font-size:11px;color:#1A7C44;background:#EDFAF2;border:1px solid #B2EAC8;border-radius:8px;padding:8px 12px;font-family:DM Mono,monospace'>
            ✅ Order: {order_str} &nbsp;|&nbsp; S={m_val} &nbsp;|&nbsp; Train: {n_train} minggu &nbsp;|&nbsp;
            MAPE: {mape_val}% &nbsp;|&nbsp; AIC: {aic_val} &nbsp;|&nbsp;
            <span style='color:#94a3b8'>Trained: {trained_at}</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:11px;color:#1A7C44;background:#EDFAF2;border:1px solid #B2EAC8;border-radius:8px;padding:8px 12px'>✅ Model tersedia. Detail tidak ditemukan di DB.</div>", unsafe_allow_html=True)

except Exception as e:
    import traceback
    st.error(f"❌ Error di section Model Status: {e}")
    with st.expander("Detail error"):
        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FORECAST — DB cache first, hitung ulang hanya jika perlu
# ══════════════════════════════════════════════════════════════════════════════
FC_START            = "2026-01-01"
fc_df               = pd.DataFrame()
using_real          = False
forecast_from_cache = False
fc_horizon_months   = 6
fc_horizon_weeks    = int(fc_horizon_months * 4.33)

# ── FIX: Seluruh blok forecast dibungkus try/except ──────────────────────────
try:
    if SARIMA_OK and model_trained:
        # Step 1: load dari tabel sarima_forecasts
        if sarima_load_fc_db is not None and callable(sarima_load_fc_db):
            try:
                cached_fc = sarima_load_fc_db(sel_villa, year=2026)
                if cached_fc is not None and not cached_fc.empty:
                    cached_fc["predicted"] = (cached_fc["predicted_occupancy"] * 100).round(1)
                    cached_fc["lower"]     = (cached_fc["lower_bound"] * 100).round(1)
                    cached_fc["upper"]     = (cached_fc["upper_bound"] * 100).round(1)
                    cached_fc["month_num"] = cached_fc.index.month
                    cached_fc["month"]     = cached_fc.index.strftime("%b %Y")
                    fc_reset = cached_fc.reset_index()
                    fc_df = (
                        fc_reset.groupby(["month","month_num"])
                        .agg(predicted=("predicted","mean"), lower=("lower","min"), upper=("upper","max"))
                        .reset_index()
                    )
                    fc_df["date"] = pd.to_datetime([f"2026-{m:02d}-01" for m in fc_df["month_num"]])
                    fc_df         = fc_df.sort_values("month_num").reset_index(drop=True)
                    using_real    = True
                    forecast_from_cache = True
            except Exception as ex:
                st.warning(f"Gagal load forecast dari DB: {ex}")

        # Step 2: belum di-cache → hitung dari model
        if fc_df.empty and sarima_forecast is not None and callable(sarima_forecast):
            try:
                raw_fc = sarima_forecast(sel_villa, horizon=fc_horizon_weeks,
                                         target_end_date="2026-06-30", use_cache=False)
                if raw_fc is not None and not raw_fc.empty and "error" not in raw_fc.columns:
                    raw_fc["predicted"] = (raw_fc["predicted_occupancy"] * 100).round(1)
                    raw_fc["lower"]     = (raw_fc["lower_bound"] * 100).round(1)
                    raw_fc["upper"]     = (raw_fc["upper_bound"] * 100).round(1)
                    raw_fc["month_num"] = raw_fc.index.month
                    raw_fc["month"]     = raw_fc.index.strftime("%b %Y")
                    raw_fc = raw_fc.reset_index()
                    fc_df = (
                        raw_fc.groupby(["month","month_num"])
                        .agg(predicted=("predicted","mean"), lower=("lower","min"), upper=("upper","max"))
                        .reset_index()
                    )
                    fc_df["date"] = pd.to_datetime([f"2026-{m:02d}-01" for m in fc_df["month_num"]])
                    fc_df         = fc_df.sort_values("month_num").reset_index(drop=True)
                    using_real    = True
                else:
                    err_msg = raw_fc["error"].iloc[0] if raw_fc is not None and "error" in raw_fc.columns else "unknown"
                    st.warning(f"Forecast SARIMA gagal: {err_msg}. Menggunakan Seasonal Naive.")
            except Exception as ex:
                st.warning(f"Forecast error: {ex}. Menggunakan Seasonal Naive.")

except Exception as e:
    import traceback
    st.warning(f"Error generate forecast SARIMA: {e}. Fallback ke Seasonal Naive.")

# ── Seasonal Naive fallback ───────────────────────────────────────────────────
if fc_df.empty:
    try:
        vd_naive  = df_occ[df_occ["villa_name"] == sel_villa].copy()
        vd_weekly = vd_naive.set_index("date").resample("W")["occupancy_pct"].mean()
        avg_by_m  = vd_weekly.groupby(vd_weekly.index.month).mean().to_dict()
        overall   = vd_weekly.mean() if len(vd_weekly) > 0 else 70.0
        rows_naive = []
        for d in pd.date_range(FC_START, end="2026-06-30", freq="W"):
            pred = float(np.clip(avg_by_m.get(d.month, overall), 0, 100))
            ci   = pred * 0.10
            rows_naive.append({"date": d, "month": d.strftime("%b %Y"), "month_num": d.month,
                                "predicted": round(pred,1), "upper": round(min(100,pred+ci),1), "lower": round(max(0,pred-ci),1)})
        fc_naive = pd.DataFrame(rows_naive)
        fc_df = fc_naive.groupby(["month","month_num"]).agg({"predicted":"mean","lower":"min","upper":"max"}).reset_index()
        fc_df["date"] = pd.to_datetime([f"2026-{m:02d}-01" for m in fc_df["month_num"]])
        fc_df = fc_df.sort_values("month_num").reset_index(drop=True)
    except Exception as e:
        st.error(f"Seasonal Naive fallback juga gagal: {e}")

# ── Data aktual 2026 ──────────────────────────────────────────────────────────
actual_2026_raw = df_occ[(df_occ["villa_name"] == sel_villa) & (df_occ["year"] == 2026)].copy()
actual_monthly  = pd.DataFrame()
if not actual_2026_raw.empty:
    try:
        actual_monthly = (
            actual_2026_raw.groupby("month_num")["occupancy_pct"].mean().reset_index()
            .rename(columns={"occupancy_pct": "actual"})
        )
        actual_monthly["actual"] = actual_monthly["actual"].round(1)
        actual_monthly["month"]  = actual_monthly["month_num"].apply(
            lambda m: pd.Timestamp(f"2026-{m:02d}-01").strftime("%b %Y")
        )
    except Exception as e:
        st.warning(f"Gagal proses data aktual 2026: {e}")

has_actual  = not actual_monthly.empty
model_label = "SARIMA" if using_real else "Seasonal Naive"
cache_label = " · 💾 dari DB cache" if forecast_from_cache else " · 🔄 dihitung baru"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CHART PREDIKSI VS AKTUAL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='rule'>", unsafe_allow_html=True)

try:
    if has_actual:
        st.markdown(f"""<div class='section-label'>Prediksi vs Aktual Okupansi 2026</div>
    <div style='background:#EEF3FF;border:1px solid #C5D7FF;border-radius:10px;padding:10px 16px;margin-bottom:14px;font-size:12px;color:#1d4ed8'>
      ✅ <b>Data aktual tersedia</b> — {len(actual_monthly)} bulan &nbsp;·&nbsp; Garis oranye = realisasi &nbsp;·&nbsp; Garis biru = prediksi {model_label}{cache_label}
    </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='section-label'>Prediksi Okupansi 2026</div>
    <div style='background:#FFF8EC;border:1px solid #FFD88A;border-radius:10px;padding:10px 16px;margin-bottom:14px;font-size:12px;color:#A05C00'>
      ⏳ <b>Belum ada data aktual 2026</b> &nbsp;·&nbsp; Model: {model_label}{cache_label}
    </div>""", unsafe_allow_html=True)

    if not fc_df.empty:
        fig_occ = go.Figure()
        if show_ci:
            r_, g_, b_ = safe_hex(vc)
            fig_occ.add_trace(go.Scatter(
                x=list(fc_df["month"]) + list(fc_df["month"][::-1]),
                y=list(fc_df["upper"]) + list(fc_df["lower"][::-1]),
                fill="toself", fillcolor=f"rgba({r_},{g_},{b_},0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="CI 95%", hoverinfo="skip", showlegend=True,
            ))
        fig_occ.add_trace(go.Scatter(
            x=fc_df["month"], y=fc_df["predicted"], mode="lines+markers",
            name=f"Prediksi {model_label}", line=dict(color=vc, width=3),
            marker=dict(size=10, symbol="diamond", color=vc, line=dict(color="#F7F7F5", width=2)),
            hovertemplate="<b>%{x}</b><br>Prediksi: %{y:.1f}%<extra></extra>",
        ))
        if has_actual:
            fig_occ.add_trace(go.Scatter(
                x=actual_monthly["month"], y=actual_monthly["actual"],
                mode="lines+markers", name="Aktual 2026",
                line=dict(color="#f97316", width=2.8, dash="dot"),
                marker=dict(size=10, symbol="circle", color="#f97316", line=dict(color="#fff", width=2)),
                hovertemplate="<b>%{x}</b><br>Aktual: %{y:.1f}%<extra></extra>",
            ))
        fig_occ.update_layout(
            **LAYOUT, height=380, margin=dict(l=0, r=0, t=60, b=40),
            title=dict(text=f"{'Prediksi vs Aktual' if has_actual else 'Prediksi'} Okupansi {sel_villa} — 2026",
                       font=dict(size=16, color="#111", family="DM Serif Display"), x=0, xanchor="left"),
            legend=dict(orientation="h", y=1.15, x=0, xanchor="left", font_size=12, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, tickangle=-35, tickfont_size=11, linecolor="#E8E8E5",
                       type="category", categoryorder="array", categoryarray=list(fc_df["month"])),
            yaxis=dict(showgrid=True, gridcolor="#F0F0EE", ticksuffix="%", range=[0,110], tickfont_size=11),
        )
        st.plotly_chart(fig_occ, use_container_width=True)
    else:
        st.info("Tidak ada data forecast untuk ditampilkan.")

except Exception as e:
    import traceback
    st.error(f"❌ Error render chart prediksi: {e}")
    with st.expander("Detail error"):
        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TABEL PREDIKSI PER BULAN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Prediksi Okupansi Per Bulan 2026</div>", unsafe_allow_html=True)

try:
    m_info = sel_db_row.get("m_used", 52) if sel_db_row else 52
    st.markdown(f"""<div style='background:#F5F7FF;border:1px solid #D8E0FF;border-radius:10px;padding:12px 16px;margin-bottom:16px;font-size:12px;color:#3D5BC0'>
      <b>📐 Model:</b> {model_label} &nbsp;|&nbsp; <b>Vila:</b> {sel_villa} &nbsp;|&nbsp;
      <b>S = {m_info}</b> &nbsp;|&nbsp; <b>Horizon:</b> {fc_horizon_months} bulan &nbsp;|&nbsp;
      {'💾 Forecast dari DB cache' if forecast_from_cache else '🔄 Forecast dihitung baru'}
    </div>""", unsafe_allow_html=True)

    if not fc_df.empty:
        tbl_df = fc_df.copy()
        if has_actual:
            tbl_df = pd.merge(tbl_df, actual_monthly[["month_num","actual"]], on="month_num", how="left")
        else:
            tbl_df["actual"] = np.nan

        rows_html = ""
        cards_html = ""
        for _, row in tbl_df.iterrows():
            occ_pred       = float(row["predicted"])
            month_label    = row["month"]
            actual_val     = row.get("actual", np.nan)
            has_row_actual = not pd.isna(actual_val)
            oc    = "#1A7C44" if occ_pred >= 80 else "#A05C00" if occ_pred >= 50 else "#C0392B"
            bar_w = int(occ_pred)

            if has_row_actual:
                err_val   = float(actual_val) - occ_pred
                err_color = "#1A7C44" if err_val >= 0 else "#C0392B"
                actual_cell = (f"<span style='font-family:DM Mono,monospace;font-size:14px;font-weight:700;color:#f97316'>{actual_val:.1f}%</span>"
                               f"&nbsp;<span style='font-size:10px;color:{err_color};font-family:DM Mono,monospace'>({err_val:+.1f}pp)</span>")
            else:
                actual_cell = "<span style='color:#cbd5e1;font-size:12px;font-family:DM Mono,monospace'>—</span>"

            rows_html += f"""<tr>
              <td style='width:22%'><b style='font-size:15px'>{month_label}</b></td>
              <td style='width:48%'><div style='display:flex;align-items:center;gap:14px'>
                <span style='font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:{oc};min-width:72px'>{occ_pred:.1f}%</span>
                <div style='flex:1'><div style='background:#F5F5F3;border-radius:100px;height:8px;overflow:hidden'>
                  <div style='background:{oc};width:{bar_w}%;height:100%;border-radius:100px'></div></div></div></div></td>
              <td style='width:30%'>{actual_cell}</td></tr>"""

            cards_html += f"""<div class='forecast-card'>
              <div class='forecast-card-header'><b style='font-size:16px'>{month_label}</b>
                <div style='text-align:right'><span style='font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:{oc}'>{occ_pred:.1f}%</span>
                {"<br>" + actual_cell if has_row_actual else ""}</div></div>
              <div><div style='background:#F5F5F3;border-radius:100px;height:8px;overflow:hidden'>
                <div style='background:{oc};width:{bar_w}%;height:100%;border-radius:100px'></div></div></div></div>"""

        actual_note = "ter-upload" if has_actual else "belum ada data"
        st.markdown(f"""
        <div class='forecast-table-wrapper'>
          <table class='forecast-table'><thead><tr>
            <th style='width:22%'>Bulan</th>
            <th style='width:48%'>Prediksi Okupansi <span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:9px'>{model_label}</span></th>
            <th style='width:30%'>Aktual 2026 <span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:9px'>{actual_note}</span></th>
          </tr></thead><tbody>{rows_html}</tbody></table>
        </div>
        <div class='forecast-cards-wrapper'>{cards_html}</div>
        """, unsafe_allow_html=True)

except Exception as e:
    import traceback
    st.error(f"❌ Error render tabel prediksi: {e}")
    with st.expander("Detail error"):
        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUASI AKURASI
# ══════════════════════════════════════════════════════════════════════════════
if has_actual and not fc_df.empty:
    try:
        eval_df = pd.merge(fc_df[["month_num","month","predicted"]],
                           actual_monthly[["month_num","actual","month"]],
                           on="month_num", how="inner", suffixes=("_fc","_act"))
        eval_df["month"]     = eval_df["month_fc"]
        eval_df              = eval_df.drop(columns=["month_fc","month_act"], errors="ignore")
        eval_df["error"]     = (eval_df["actual"] - eval_df["predicted"]).round(2)
        eval_df["abs_error"] = eval_df["error"].abs()
        eval_df["ape"]       = (eval_df["abs_error"] / eval_df["actual"].replace(0, np.nan) * 100).round(2)

        mae_live  = eval_df["abs_error"].mean()
        mape_live = eval_df["ape"].mean()
        rmse_live = float(np.sqrt((eval_df["error"] ** 2).mean()))
        bias      = eval_df["error"].mean()

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>📐 Evaluasi Akurasi — Prediksi vs Aktual 2026</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:12px;color:#888;margin-bottom:16px'>Dihitung dari <b>{len(eval_df)} bulan</b> yang sudah tersedia data aktualnya.</p>", unsafe_allow_html=True)

        ka, kb, kc, kd = st.columns(4)
        for col, label, val, note, color in [
            (ka, "MAPE Aktual", f"{mape_live:.1f}%", "< 10% Sangat Baik · < 20% Baik",
             "#1A7C44" if mape_live < 10 else "#A05C00" if mape_live < 20 else "#C0392B"),
            (kb, "MAE",  f"{mae_live:.1f}pp",  "Mean Absolute Error",
             "#1A7C44" if mae_live < 5 else "#A05C00" if mae_live < 15 else "#C0392B"),
            (kc, "RMSE", f"{rmse_live:.1f}pp", "Root Mean Squared Error",
             "#1A7C44" if rmse_live < 5 else "#A05C00" if rmse_live < 15 else "#C0392B"),
            (kd, "Bias", f"{bias:+.1f}pp", "+ = under-forecast · − = over-forecast",
             "#1d4ed8" if abs(bias) < 3 else "#A05C00"),
        ]:
            with col:
                st.markdown(f"""<div class='exec-metric'><div class='exec-metric-label'>{label}</div>
                  <div class='exec-metric-value' style='color:{color};font-size:26px'>{val}</div>
                  <div class='exec-metric-delta'>{note}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        bar_colors = ["#1A7C44" if e >= 0 else "#C0392B" for e in eval_df["error"]]
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(x=eval_df["month"], y=eval_df["error"], marker_color=bar_colors, marker_line_width=0,
            customdata=eval_df[["actual","predicted","ape"]].values,
            hovertemplate="<b>%{x}</b><br>Aktual: %{customdata[0]:.1f}%<br>Prediksi: %{customdata[1]:.1f}%<br>Error: %{y:+.1f}pp<br>APE: %{customdata[2]:.1f}%<extra></extra>"))
        fig_err.add_hline(y=0, line_color="#334155", line_width=1.5)
        for _, row in eval_df.iterrows():
            fig_err.add_annotation(x=row["month"], y=row["error"] + (2.5 if row["error"] >= 0 else -4.5),
                text=f"{row['ape']:.1f}%", showarrow=False, font=dict(size=9, color="#334155", family="DM Mono"))
        fig_err.update_layout(**LAYOUT, height=300, margin=dict(l=0, r=0, t=60, b=40), showlegend=False,
            title=dict(text="Error per Bulan (Aktual − Prediksi)", font=dict(size=14, color="#111", family="DM Serif Display"), x=0),
            xaxis=dict(showgrid=False, tickangle=-35, tickfont_size=11, linecolor="#E8E8E5", type="category",
                       categoryorder="array", categoryarray=list(eval_df["month"])),
            yaxis=dict(showgrid=True, gridcolor="#F0F0EE", ticksuffix="pp", tickfont_size=11, zeroline=False))
        st.plotly_chart(fig_err, use_container_width=True)

        with st.expander("📋 Detail tabel evaluasi", expanded=False):
            eval_display = eval_df[["month","predicted","actual","error","ape"]].copy()
            eval_display.columns = ["Bulan","Prediksi (%)","Aktual (%)","Error (pp)","APE (%)"]
            def style_error(v):
                if not isinstance(v, (int, float)): return ""
                if v > 0: return "color: #1A7C44; font-weight: 600"
                if v < 0: return "color: #C0392B; font-weight: 600"
                return ""
            st.dataframe(eval_display.style
                .format({"Prediksi (%)":"{:.1f}","Aktual (%)":"{:.1f}","Error (pp)":"{:+.1f}","APE (%)":"{:.1f}%"})
                .applymap(style_error, subset=["Error (pp)"]), use_container_width=True, hide_index=True)

    except Exception as e:
        import traceback
        st.error(f"❌ Error render evaluasi akurasi: {e}")
        with st.expander("Detail error"):
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SUMMARY SEMUA VILA
# ══════════════════════════════════════════════════════════════════════════════
try:
    trained_villas = []
    if SARIMA_OK and sarima_exists is not None and callable(sarima_exists):
        for vn in villa_names:
            try:
                if sarima_exists(vn):
                    trained_villas.append(vn)
            except Exception:
                pass

    if trained_villas:
        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Evaluasi Model SARIMA — Summary Semua Vila</div>", unsafe_allow_html=True)

        def fmt_v(v, unit="", dec=2):
            if v is None: return "—"
            try:
                f = float(v)
                return "—" if np.isnan(f) else f"{f:.{dec}f}{unit}"
            except: return "—"

        def mc(v):
            try: v = float(v)
            except: return "#888"
            return "#1A7C44" if v < 10 else "#A05C00" if v < 20 else "#C0392B"

        def rc(v):
            try: v = float(v)
            except: return "#888"
            return "#1A7C44" if v < 5 else "#A05C00" if v < 15 else "#C0392B"

        rows_acc = ""
        for vn in villa_names:
            is_trained = vn in trained_villas
            vn_color   = get_color(vn)
            area       = VILLA_INSIGHTS.get(vn, {}).get("area", "")
            dot        = f"<span class='villa-dot' style='background:{vn_color}'></span>"
            db         = get_sarima_db_row(vn, sarima_summary_db)

            if not is_trained or not db:
                rows_acc += f"""<tr style='opacity:0.40'>
                  <td>{dot}<b>{vn}</b><br><span style='font-size:10px;color:#aaa;font-family:DM Mono,monospace'>{area}</span></td>
                  <td colspan='4' style='font-size:11px;color:#bbb;font-family:DM Mono,monospace;text-align:center'>⏳ Belum dilatih</td></tr>"""
                continue

            mape_v = db.get("mape"); rmse_v = db.get("rmse"); aic_v = db.get("aic")
            m_val  = db.get("m_used", 52); n_tr = db.get("n_train","—"); n_cy = db.get("n_cycles","—")
            tr_at  = str(db.get("trained_at",""))[:10]
            ao     = db.get("arima_order",""); so = db.get("seasonal_order","")

            import re
            m_sub = {"4":"₄","12":"₁₂","26":"₂₆","52":"₅₂"}.get(str(m_val), f"[{m_val}]")
            so_disp = re.sub(r",(\d+)\)$", f",{m_sub})", str(so)) if so else ""
            order_str = f"{ao}{so_disp}" if ao else "—"

            if ")(" in order_str:
                p1, p2     = order_str.split(")(")
                order_html = (f"<span style='font-size:13px;font-weight:700;font-family:DM Mono,monospace;color:#7C3AED'>{p1})</span>"
                               f"<span style='font-size:12px;font-weight:600;font-family:DM Mono,monospace;color:#A855F7'>({p2}</span>")
            else:
                order_html = f"<span style='font-size:13px;font-weight:700;font-family:DM Mono,monospace;color:#7C3AED'>{order_str}</span>"

            aic_disp = fmt_v(round(float(aic_v)) if aic_v else None, "", 0)

            rows_acc += f"""<tr>
              <td>{dot}<b>{vn}</b>
                <br><span style='font-size:10px;color:#aaa;font-family:DM Mono,monospace'>{area}</span>
                <br><span style='font-size:9px;color:#cbd5e1;font-family:DM Mono,monospace'>trained {tr_at}</span>
              </td>
              <td><span style='font-size:17px;font-weight:700;font-family:DM Mono,monospace;color:{mc(mape_v)}'>{fmt_v(mape_v,"%",2)}</span></td>
              <td><span style='font-size:15px;font-weight:700;font-family:DM Mono,monospace;color:{rc(rmse_v)}'>{fmt_v(rmse_v,"",2)}</span></td>
              <td><span style='font-size:13px;font-family:DM Mono,monospace;color:#3D6BE8'>{aic_disp}</span></td>
              <td>{order_html}
                <br><span style='font-size:9px;color:#94a3b8;font-family:DM Mono,monospace'>S={m_val} · {n_tr} minggu · {n_cy} siklus</span>
              </td></tr>"""

        st.markdown(f"""
        <table class='summary-acc-table'>
          <thead><tr>
            <th>Vila</th>
            <th>MAPE<br><span style='font-weight:400;font-size:8px;text-transform:none;letter-spacing:0'>Error rata-rata %</span></th>
            <th>RMSE<br><span style='font-weight:400;font-size:8px;text-transform:none;letter-spacing:0'>Error kuadrat</span></th>
            <th>AIC<br><span style='font-weight:400;font-size:8px;text-transform:none;letter-spacing:0'>Fit model</span></th>
            <th>Model SARIMA<br><span style='font-weight:400;font-size:8px;text-transform:none;letter-spacing:0'>(p,d,q)(P,D,Q)[S]</span></th>
          </tr></thead>
          <tbody>{rows_acc}</tbody>
        </table>""", unsafe_allow_html=True)

except Exception as e:
    import traceback
    st.error(f"❌ Error render summary SARIMA: {e}")
    with st.expander("Detail error"):
        st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ANALISIS DESKRIPTIF ADR & OKUPANSI
# ══════════════════════════════════════════════════════════════════════════════
def build_monthly_data(villa_name: str, year_filter=None):
    occ = df_occ[df_occ["villa_name"] == villa_name].copy()
    if year_filter: occ = occ[occ["year"].isin(year_filter)]
    occ["period"] = occ["date"].dt.to_period("M").astype(str)

    monthly_occ = occ.groupby("period")["occupancy_pct"].mean().reset_index().rename(columns={"occupancy_pct":"occ"})

    vi        = VILLA_INSIGHTS.get(villa_name, {})
    corr_info = {"r": 0, "live": False, "elastisitas": vi.get("elastisitas","Semi-Elastis"), "area": vi.get("area","")}

    if df_fin is None or df_fin.empty or "avg_daily_revenue" not in df_fin.columns:
        monthly_occ["adr_m"] = np.nan
        return monthly_occ, corr_info, pd.DataFrame()

    fin = df_fin[df_fin["villa_name"] == villa_name].copy()
    if year_filter: fin = fin[fin["year"].isin(year_filter)]
    fin_clean = fin[fin["for_modeling"] == 1].copy() if "for_modeling" in fin.columns else fin[fin["avg_daily_revenue"] > 0].copy()

    if fin_clean.empty:
        monthly_occ["adr_m"] = np.nan
        return monthly_occ, corr_info, pd.DataFrame()

    fin_clean["period"] = fin_clean["date"].dt.to_period("M").astype(str)
    monthly_adr = fin_clean.groupby("period")["avg_daily_revenue"].mean().reset_index().rename(columns={"avg_daily_revenue":"adr_m"})
    monthly_adr["adr_m"] /= 1_000_000
    monthly = pd.merge(monthly_occ, monthly_adr, on="period", how="left").sort_values("period").reset_index(drop=True)

    daily = pd.merge(occ[["date","occupancy_pct"]], fin_clean[["date","avg_daily_revenue"]], on="date", how="inner").dropna()

    if len(daily) >= 10:
        try:
            r_val, _ = stats.pearsonr(daily["occupancy_pct"], daily["avg_daily_revenue"])
            corr_info.update({
                "r": round(r_val,2), "n": len(daily), "live": True,
                "adr_median": round(fin_clean["avg_daily_revenue"].median()/1_000_000, 2),
                "adr_max":    round(fin_clean["avg_daily_revenue"].max()   /1_000_000, 2),
                "adr_min":    round(fin_clean["avg_daily_revenue"].min()   /1_000_000, 2),
                "adr_std":    round(fin_clean["avg_daily_revenue"].std()   /1_000_000, 2),
            })
        except Exception: pass

    return monthly, corr_info, daily


st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("""<div class='anim'>
  <div class='section-label'>📈 Analisis Deskriptif — ADR & Okupansi</div>
  <p style='font-size:13px;color:#666;margin-bottom:4px'>Semua angka dihitung live dari database. Elastisitas & strategi adalah konfigurasi bisnis.</p>
</div>""", unsafe_allow_html=True)

try:
    all_areas = sorted(set(VILLA_INSIGHTS.get(v,{}).get("area","") for v in villa_names if VILLA_INSIGHTS.get(v,{}).get("area","")))
    all_years = sorted(df_occ["year"].unique().tolist())

    fa, fb, fc_col = st.columns([1.5, 2.5, 2])
    with fa:
        sel_area_desc = st.multiselect("Filter Area", options=all_areas, default=all_areas, key="desc_area", placeholder="Semua area")
    with fb:
        fv = [v for v in villa_names if VILLA_INSIGHTS.get(v,{}).get("area","") in (sel_area_desc or all_areas)]
        sel_villas_desc = st.multiselect("Filter Vila", options=fv, default=fv, key="desc_villa", placeholder="Semua vila")
    with fc_col:
        sel_years_desc = st.multiselect("Filter Tahun", options=all_years, default=all_years, key="desc_year", placeholder="Semua tahun")

    display_villas = sel_villas_desc if sel_villas_desc else fv
    year_filter    = sel_years_desc  if sel_years_desc  else None

    if not display_villas:
        st.info("Pilih minimal satu vila.")
    else:
        for tab, vn in zip(st.tabs([v.replace(" Villas","") for v in display_villas]), display_villas):
            with tab:
                try:
                    vc4 = get_color(vn)
                    r_hex, g_hex, b_hex = safe_hex(vc4)
                    monthly, cd2, daily_merged = build_monthly_data(vn, year_filter=year_filter)

                    if monthly.empty:
                        st.info(f"Belum ada data untuk {vn}.")
                        continue

                    if "adr_m" not in monthly.columns: monthly["adr_m"] = np.nan

                    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_dual.add_trace(go.Bar(x=monthly["period"], y=monthly["occ"], name="Okupansi (%)",
                        marker_color=f"rgba({r_hex},{g_hex},{b_hex},0.55)",
                        marker_line_color=f"rgba({r_hex},{g_hex},{b_hex},0.85)", marker_line_width=1.2,
                        hovertemplate="<b>%{x}</b><br>Ocupansi: %{y:.1f}%<extra></extra>"), secondary_y=True)

                    adr_data = monthly.dropna(subset=["adr_m"])
                    if not adr_data.empty:
                        fig_dual.add_trace(go.Scatter(x=adr_data["period"], y=adr_data["adr_m"], name="ADR (Juta Rupiah)",
                            mode="lines+markers", line=dict(color="#f97316", width=2.5),
                            marker=dict(size=7, color="#f97316"), hovertemplate="<b>%{x}</b><br>ADR: Rp%{y:.2f} Juta<extra></extra>"), secondary_y=False)

                    r_val = cd2.get("r", 0); n_obs = cd2.get("n","—")
                    src   = "live DB" if cd2.get("live") else "tidak ada data keuangan"
                    fig_dual.update_layout(**LAYOUT, height=360, margin=dict(l=0,r=0,t=60,b=40),
                        title=dict(text=f"Tren Musiman — {vn} &nbsp;<span style='font-size:12px;color:#666'>r = {r_val:+.2f} · n={n_obs} · {src}</span>",
                                   font=dict(size=14,color="#111",family="DM Serif Display"), x=0),
                        legend=dict(orientation="h", y=1.12, x=0, font_size=11, bgcolor="rgba(0,0,0,0)"),
                        xaxis=dict(showgrid=False, tickangle=-45, tickfont_size=10, linecolor="#E8E8E5", type="category"), bargap=0.25)
                    fig_dual.update_yaxes(title_text="ADR (Juta Rupiah)", secondary_y=False, showgrid=True, gridcolor="#F0F0EE", ticksuffix=" Jt", tickfont_size=11, rangemode="tozero")
                    fig_dual.update_yaxes(title_text="Ocupansi (%)", secondary_y=True, showgrid=False, ticksuffix="%", range=[0,130], tickfont_size=11)
                    st.plotly_chart(fig_dual, use_container_width=True)

                    if not daily_merged.empty and len(daily_merged) >= 5:
                        sx = daily_merged["occupancy_pct"].values
                        sy = daily_merged["avg_daily_revenue"].values / 1_000_000
                        r_sc = cd2.get("r",0); r2 = round(r_sc**2, 3)
                        adr_med = np.median(sy)
                        pc = [f"rgba({r_hex},{g_hex},{b_hex},0.65)" if y >= adr_med else f"rgba({r_hex},{g_hex},{b_hex},0.30)" for y in sy]

                        try:
                            mc2, bc2 = np.polyfit(sx, sy, 1)
                            xl = np.linspace(sx.min(), sx.max(), 100); yl = mc2*xl+bc2; has_reg=True
                        except: has_reg=False

                        fig_sc = go.Figure()
                        fig_sc.add_trace(go.Scatter(x=sx, y=sy, mode="markers",
                            marker=dict(size=6, color=pc, line=dict(color=vc4, width=0.5)), name="Hari Observasi",
                            hovertemplate="Ocupansi: %{x:.1f}%<br>ADR: Rp%{y:.2f} Juta<extra></extra>"))
                        if has_reg:
                            fig_sc.add_trace(go.Scatter(x=xl, y=yl, mode="lines",
                                line=dict(color="#f97316", width=2.5, dash="dash"), name=f"Regresi (r={r_sc:+.2f})", hoverinfo="skip"))
                        fig_sc.add_annotation(xref="paper", yref="paper", x=0.97, y=0.97,
                            text=f"<b>r = {r_sc:+.2f}</b>  |  R² = {r2:.3f}  |  n = {len(sx)}",
                            showarrow=False, bgcolor="#fff", bordercolor=vc4, borderwidth=1,
                            font=dict(size=12, color=vc4, family="DM Mono"), align="right")
                        fig_sc.update_layout(**LAYOUT, height=320, margin=dict(l=0,r=0,t=50,b=40),
                            title=dict(text=f"Scatter: ADR vs Tingkat Okupansi — {vn}",
                                       font=dict(size=13,color="#111",family="DM Serif Display"), x=0),
                            legend=dict(orientation="h", y=1.12, x=0, font_size=11, bgcolor="rgba(0,0,0,0)"),
                            xaxis=dict(title="Ocupansi (%)", showgrid=True, gridcolor="#F0F0EE", ticksuffix="%", tickfont_size=11),
                            yaxis=dict(title="ADR (Juta Rupiah)", showgrid=True, gridcolor="#F0F0EE", ticksuffix=" Jt", tickfont_size=11, rangemode="tozero"))
                        st.plotly_chart(fig_sc, use_container_width=True)

                        abs_r = abs(r_sc)
                        if   abs_r >= 0.7: cs,cc,cb,cbr,ci2 = "Kuat",         "#1A7C44","#EDFAF2","#B2EAC8","🔗"
                        elif abs_r >= 0.4: cs,cc,cb,cbr,ci2 = "Moderat",      "#2563EB","#EEF3FF","#C5D7FF","〰️"
                        elif abs_r >= 0.2: cs,cc,cb,cbr,ci2 = "Lemah",        "#A05C00","#FFF8EC","#FFD88A","📉"
                        else:              cs,cc,cb,cbr,ci2 = "Sangat Lemah", "#64748b","#f1f5f9","#e2e8f0","➖"

                        direction = "positif" if r_sc >= 0 else "negatif"
                        adr_med_d = cd2.get("adr_median",0); adr_hi_d = cd2.get("adr_max",0); adr_lo_d = cd2.get("adr_min",0)
                        elastisitas = cd2.get("elastisitas", VILLA_INSIGHTS.get(vn,{}).get("elastisitas","Semi-Elastis"))

                        st.markdown(f"""
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px'>
                          <div style='background:{cb};border:1px solid {cbr};border-radius:10px;padding:14px;border-top:3px solid {cc}'>
                            <div style='font-family:DM Mono,monospace;font-size:9px;color:{cc};text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>{ci2} Korelasi Pearson</div>
                            <div style='font-size:26px;font-weight:800;font-family:DM Mono,monospace;color:{cc};margin-bottom:4px'>r = {r_sc:+.2f}</div>
                            <div style='font-size:12px;font-weight:700;color:{cc};margin-bottom:4px'>{cs} · {elastisitas}</div>
                            <div style='font-size:11px;color:#555;line-height:1.5'>
                              Arah: {direction} &nbsp;·&nbsp; R² = {r2:.3f} ({r2*100:.1f}%)<br>
                              n = {len(sx)} hari · <i>dihitung live dari DB</i>
                            </div>
                          </div>
                          <div style='background:#fff;border:1px solid #E8E8E5;border-radius:10px;padding:14px;border-top:3px solid {vc4}'>
                            <div style='font-family:DM Mono,monospace;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Profil Harga · live DB</div>
                            <div style='font-size:22px;font-weight:700;font-family:DM Mono,monospace;color:{vc4};margin-bottom:4px'>Rp{adr_med_d:.2f} Juta</div>
                            <div style='font-size:11px;color:#888'>Median ADR (hari ada revenue)</div>
                            <div style='font-size:11px;color:#aaa;margin-top:4px'>Range: Rp{adr_lo_d:.2f} – Rp{adr_hi_d:.2f} Juta</div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Data keuangan tidak cukup untuk analisis korelasi.")

                except Exception as e:
                    import traceback
                    st.error(f"❌ Error render tab {vn}: {e}")
                    with st.expander("Detail error"):
                        st.code(traceback.format_exc())

except Exception as e:
    import traceback
    st.error(f"❌ Error section analisis deskriptif: {e}")
    with st.expander("Detail error"):
        st.code(traceback.format_exc())

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""<div style='text-align:center;padding:28px 0 8px;color:#ccc;font-size:11px;
    font-family:DM Mono,monospace;letter-spacing:.08em'>
  VILLAS R US ANALYTICS · STRATEGI OKUPANSI · SARIMA FORECAST · 2026
</div>""", unsafe_allow_html=True)