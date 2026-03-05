import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from utils.auth import get_cookie_manager, set_session, load_from_cookie, logout
from utils.sidebar import render_sidebar

# ─── WAJIB DI PALING ATAS ────────────────────────────────────────────────────
cookies = get_cookie_manager()
if not cookies.ready():
    st.stop()

# ─── AUTH GUARD ──────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    user_data = load_from_cookie(cookies)
    if user_data:
        set_session(user_data)
    else:
        st.switch_page("streamlit_app.py")
        st.stop()

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — Villas R Us",
    page_icon="🏝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# SIDEBAR
render_sidebar(cookies)

# ─── LOGOUT HANDLER ──────────────────────────────────────────────────────────
if st.session_state.get("do_logout"):
    logout(cookies)
    st.switch_page("streamlit_app.py")
    st.stop()

from database import get_occupancy_data, get_financial_data, get_villas

SARIMA_OK = False
TRAIN_OK = False
SARIMA_ERR = ""
sarima_forecast = None
sarima_exists = None
sarima_train = None
sarima_get_meta = None

try:
    from utils.sarima_engine import forecast as sarima_forecast
    from utils.sarima_engine import model_exists as sarima_exists
    SARIMA_OK = True
except Exception as e:
    SARIMA_ERR = str(e)

try:
    from utils.sarima_engine import train_and_save as sarima_train
    TRAIN_OK = True
except Exception:
    pass

try:
    from utils.sarima_engine import get_meta as sarima_get_meta
except Exception:
    pass

st.markdown(
    """
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
  .model-badge.ok  { background: #dcfce7; border: 1px solid #bbf7d0; color: #166534; }
  .model-badge.warn{ background: #fef9c3; border: 1px solid #fde68a; color: #854d0e; }
  .model-badge.err { background: #fee2e2; border: 1px solid #fecaca; color: #991b1b; }
  .pill { display: inline-block; font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 600; padding: 3px 10px; border-radius: 100px; }
  .pill-green { background: #dcfce7; border: 1px solid #bbf7d0; color: #166534; }
  .pill-amber { background: #fef9c3; border: 1px solid #fde68a; color: #854d0e; }
  .pill-grey  { background: #f1f5f9; border: 1px solid #e2e8f0;  color: #475569; }
  .pill-red   { background: #fee2e2; border: 1px solid #fecaca;  color: #991b1b; }
  .pill-blue  { background: #eff6ff; border: 1px solid #bfdbfe;  color: #1d4ed8; }
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
  .forecast-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(3,105,161,0.05); }
  .forecast-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #f1f5f9; }
  .forecast-card-body { font-size: 11px; line-height: 1.6; }
  .forecast-table-wrapper { display: block; }
  .forecast-cards-wrapper { display: none; }
  @media (max-width: 768px) {
    .forecast-table-wrapper { display: none; }
    .forecast-cards-wrapper { display: block; }
    .page-title { font-size: 20px; }
    .exec-metric { margin-bottom: 12px; }
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
</style>
""",
    unsafe_allow_html=True,
)

# ── Villa static config ───────────────────────────────────────────────────────
VILLA_INSIGHTS = {
    "Briana Villas": {
        "elastisitas": "Semi-Elastis", "color": "#3D6BE8", "area": "Canggu",
        "price_floor": 1.2, "price_ceiling": 3.5, "naik_pct": 15, "turun_pct": 10,
        "elasticity_factor": 0.15,
        "strategi_peak": "Naikkan rate saat occupancy >85%. Tutup diskon.",
        "strategi_low": "Tawarkan paket 3-malam dengan diskon 10%. Early bird promo.",
    },
    "Castello Villas": {
        "elastisitas": "Semi-Elastis", "color": "#7c3aed", "area": "Canggu",
        "price_floor": 1.0, "price_ceiling": 3.0, "naik_pct": 12, "turun_pct": 12,
        "elasticity_factor": 0.15,
        "strategi_peak": "Konsistensi pricing. Batasi OTA commission di peak.",
        "strategi_low": "Flash sale 12% + paket F&B bundling.",
    },
    "Elina Villas": {
        "elastisitas": "Semi-Elastis", "color": "#059669", "area": "Canggu",
        "price_floor": 3.0, "price_ceiling": 3.5, "naik_pct": 8, "turun_pct": 5,
        "elasticity_factor": 0.12,
        "strategi_peak": "Pertahankan harga premium Rp3-3.5M. Prioritas direct booking.",
        "strategi_low": "Fokus high season saja. Tutup operasional off-season.",
    },
    "Isola Villas": {
        "elastisitas": "Elastis", "color": "#db2777", "area": "Canggu",
        "price_floor": 1.5, "price_ceiling": 4.0, "naik_pct": 20, "turun_pct": 15,
        "elasticity_factor": 0.25,
        "strategi_peak": "Naikkan agresif +20%. Weekend packages. Minimum stay 2 malam.",
        "strategi_low": "Flash sale 15%. Paket mid-week spesial.",
    },
    "Eindra Villas": {
        "elastisitas": "Inelastis", "color": "#d97706", "area": "Seminyak",
        "price_floor": 4.0, "price_ceiling": 7.7, "naik_pct": 5, "turun_pct": 3,
        "elasticity_factor": 0.05,
        "strategi_peak": "Pertahankan Rp4-5M. Potensi naik tipis +5%.",
        "strategi_low": "Tawarkan kontrak korporat & long-stay. Jangan turun drastis.",
    },
    "Esha Villas": {
        "elastisitas": "Semi-Elastis", "color": "#b45309", "area": "Seminyak",
        "price_floor": 1.3, "price_ceiling": 4.3, "naik_pct": 18, "turun_pct": 12,
        "elasticity_factor": 0.18,
        "strategi_peak": "Terapkan Rp3-4M di Mei-Oktober. Rate optimizer aktif.",
        "strategi_low": "Turunkan ke Rp1.5-2M Nov-Apr. Paket liburan sekolah.",
    },
    "Ozamiz Villas": {
        "elastisitas": "Inelastis", "color": "#9333ea", "area": "Seminyak",
        "price_floor": 2.0, "price_ceiling": 5.0, "naik_pct": 4, "turun_pct": 2,
        "elasticity_factor": 0.04,
        "strategi_peak": "Fokus long-stay & kontrak korporat.",
        "strategi_low": "Pertahankan harga kontrak. Jangan diskon.",
    },
}

SARIMA_META_STATIC = {
    "Briana Villas":  {"order": "(0,1,3)(1,0,1)₅₂", "mape": 18.88, "rmse": 39.39, "aic": 10268},
    "Castello Villas":{"order": "(1,1,2)(2,0,1)₅₂", "mape": 2.86,  "rmse": 18.07, "aic": 9276},
    "Elina Villas":   {"order": "(2,0,2)(0,0,0)₅₂", "mape": 19.98, "rmse": 38.45, "aic": 10302},
    "Isola Villas":   {"order": "(3,1,2)(2,0,1)₅₂", "mape": 5.63,  "rmse": 18.96, "aic": 10416},
    "Eindra Villas":  {"order": "(1,0,0)(0,0,0)₅₂", "mape": 3.73,  "rmse": 14.94, "aic": 9714},
    "Esha Villas":    {"order": "(2,0,1)(2,0,0)₅₂", "mape": 1.33,  "rmse": 1.34,  "aic": 8999},
    "Ozamiz Villas":  {"order": "(2,0,0)(2,0,0)₅₂", "mape": 1.33,  "rmse": 1.34,  "aic": 8999},
}

ADR_STATIC = {
    "Briana Villas":  {"r": 0.48, "elastisitas": "Semi-Elastis", "median_adr": 2.0,  "adr_min": 1.0, "adr_max": 4.0,  "area": "Canggu",   "strategy": "dynamic_moderate"},
    "Castello Villas":{"r": 0.59, "elastisitas": "Semi-Elastis", "median_adr": 1.4,  "adr_min": 0.6, "adr_max": 3.0,  "area": "Canggu",   "strategy": "experimental"},
    "Elina Villas":   {"r": 0.57, "elastisitas": "Semi-Elastis", "median_adr": 3.35, "adr_min": 3.0, "adr_max": 7.0,  "area": "Canggu",   "strategy": "selective_premium"},
    "Isola Villas":   {"r": 0.86, "elastisitas": "Elastis",      "median_adr": 1.3,  "adr_min": 0.6, "adr_max": 4.0,  "area": "Canggu",   "strategy": "dynamic_active"},
    "Eindra Villas":  {"r": 0.34, "elastisitas": "Inelastis",    "median_adr": 5.2,  "adr_min": 2.9, "adr_max": 11.3, "area": "Seminyak", "strategy": "premium_fixed"},
    "Esha Villas":    {"r": 0.48, "elastisitas": "Semi-Elastis", "median_adr": 3.2,  "adr_min": 1.3, "adr_max": 4.3,  "area": "Seminyak", "strategy": "seasonal_responsive"},
    "Ozamiz Villas":  {"r": 0.22, "elastisitas": "Inelastis",    "median_adr": 3.2,  "adr_min": 0.9, "adr_max": 5.4,  "area": "Seminyak", "strategy": "corporate_contract"},
}

LAYOUT = dict(plot_bgcolor="#ffffff", paper_bgcolor="#F7F7F5", font_color="#555", font_family="DM Sans")
PEAK_MONTHS     = [6, 7, 8, 12, 1]
SHOULDER_MONTHS = [4, 5, 9, 10]
LOW_MONTHS      = [2, 3, 11]


def safe_hex(hex_color):
    h = hex_color.lstrip("#")
    if len(h) != 6:
        h = "3D6BE8"
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    return get_occupancy_data(), get_financial_data(), get_villas()


df_occ, df_fin, df_villas = load_data()

if df_occ is None or df_occ.empty:
    st.warning("Belum ada data. Silakan upload terlebih dahulu.")
    if st.button("Upload Data"):
        st.switch_page("pages/3_Upload.py")
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


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='anim'>
  <div class='page-eyebrow'>Integrated Analysis · SARIMA Forecast</div>
  <div class='page-title'>Strategi <em>Okupansi 2026</em></div>
  <div class='page-sub'>Prediksi okupansi dengan analisis komponen SARIMA dan pola historis</div>
</div>
<hr class='rule'>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EXECUTIVE SUMMARY (all villas combined)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>📊 Executive Summary 2026</div>", unsafe_allow_html=True)

hist_occ_all = df_occ["occupancy_pct"].mean() if not df_occ.empty else 70.0
peak_mask_all = df_occ["month_num"].isin(PEAK_MONTHS)
peak_avg_all = df_occ.loc[peak_mask_all, "occupancy_pct"].mean() if peak_mask_all.any() else 0.0
risk_villas_count = 0
best_villa_overall = "—"
best_occ_overall = 0.0

for vn in villa_names:
    vd_tmp = df_occ[df_occ["villa_name"] == vn]
    avg_tmp = vd_tmp["occupancy_pct"].mean() if not vd_tmp.empty else 0
    if avg_tmp < 50:
        risk_villas_count += 1
    if avg_tmp > best_occ_overall:
        best_occ_overall = avg_tmp
        best_villa_overall = vn

occ_2026_actual = df_occ[df_occ["year"] == 2026]["occupancy_pct"].mean() if (df_occ["year"] == 2026).any() else None

e1, e2, e3, e4 = st.columns(4)
with e1:
    val_disp = f"{occ_2026_actual:.1f}%" if occ_2026_actual is not None else f"{hist_occ_all:.1f}%"
    lbl_disp = "Avg Okupansi 2026 (Aktual)" if occ_2026_actual is not None else "Avg Okupansi Historis"
    st.markdown(f"""<div class='exec-metric'>
        <div class='exec-metric-label'>{lbl_disp}</div>
        <div class='exec-metric-value'>{val_disp}</div>
        <div class='exec-metric-delta'>Semua vila</div>
    </div>""", unsafe_allow_html=True)
with e2:
    st.markdown(f"""<div class='exec-metric'>
        <div class='exec-metric-label'>Peak Season Average</div>
        <div class='exec-metric-value'>{peak_avg_all:.1f}%</div>
        <div class='exec-metric-delta'>Jun–Aug, Des–Jan</div>
    </div>""", unsafe_allow_html=True)
with e3:
    rc2 = "negative" if risk_villas_count > 3 else "positive" if risk_villas_count == 0 else ""
    st.markdown(f"""<div class='exec-metric'>
        <div class='exec-metric-label'>Vila Rata-rata &lt;50%</div>
        <div class='exec-metric-value'>{risk_villas_count}/{len(villa_names)}</div>
        <div class='exec-metric-delta {rc2}'>ocupansi rata-rata &lt;50%</div>
    </div>""", unsafe_allow_html=True)
with e4:
    st.markdown(f"""<div class='exec-metric'>
        <div class='exec-metric-label'>Vila Terbaik</div>
        <div class='exec-metric-value' style='font-size:18px'>{best_villa_overall.replace(" Villas","")}</div>
        <div class='exec-metric-delta'>highest historical occupancy</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VILLA FILTER + MODEL STATUS + FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='rule'>", unsafe_allow_html=True)

# Villa selector
cc1, cc2 = st.columns([2.5, 4])
with cc1:
    sel_villa = st.selectbox("Vila", villa_names, label_visibility="collapsed", key="sv")
    st.markdown("<div style='font-size:11px;color:#aaa;margin-top:-8px'>Pilih vila</div>", unsafe_allow_html=True)
with cc2:
    show_ci = st.toggle("Confidence Interval", value=True, key="sci")

vc = get_color(sel_villa)
fc_horizon_months = 6
fc_horizon_weeks  = int(fc_horizon_months * 4.33)

# ── Model SARIMA status & train ───────────────────────────────────────────────
st.markdown("<div class='section-label'>Model SARIMA — Status &amp; Train</div>", unsafe_allow_html=True)

model_trained = SARIMA_OK and sarima_exists is not None and sarima_exists(sel_villa)
t1, t2, t3, t4 = st.columns([2.5, 1.5, 1.5, 3])

with t1:
    if not SARIMA_OK:          bcls, btxt = "err",  "Import Error"
    elif model_trained:        bcls, btxt = "ok",   "Model Tersedia"
    else:                      bcls, btxt = "warn", "Belum Dilatih"
    st.markdown(f"""
    <div style='padding:10px 0'>
      <span class='model-badge {bcls}'>{btxt}</span>
      <span style='font-size:11px;color:#aaa;margin-left:10px;font-family:DM Mono,monospace'>{sel_villa}</span>
    </div>""", unsafe_allow_html=True)

with t2:
    if TRAIN_OK and not model_trained:
        if st.button("🚀 Train Model", width='stretch', key="btn_train"):
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
        st.button("🚀 Train Model", width='stretch', disabled=True, key="btn_train_dis",
                  help="Model sudah dilatih." if model_trained else "sarima_engine tidak tersedia.")

with t3:
    if TRAIN_OK and model_trained:
        if st.button("🔄 Retrain", width='stretch', key="btn_retrain"):
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
        st.button("🔄 Retrain", width='stretch', disabled=True, key="btn_retrain_dis",
                  help="Train model terlebih dahulu.")

with t4:
    if not SARIMA_OK:
        st.markdown(f"<div style='font-size:11px;color:#C0392B;background:#FFF0F0;border:1px solid #FFC5C5;border-radius:8px;padding:8px 12px;font-family:DM Mono,monospace'>❌ Import error: {SARIMA_ERR[:120]}</div>", unsafe_allow_html=True)
    elif not model_trained:
        st.markdown("<div style='font-size:11px;color:#A05C00;background:#FFF8EC;border:1px solid #FFD88A;border-radius:8px;padding:8px 12px'>⚠️ Model belum dilatih. Klik <b>Train Model</b> untuk memulai (~3-8 menit).</div>", unsafe_allow_html=True)
    elif sarima_get_meta:
        try:
            meta      = sarima_get_meta(sel_villa)
            order_str = str(meta.get("order","?")) + str(meta.get("seasonal_order",""))
            n_train   = meta.get("n_train","?")
            aic_val   = meta.get("aic","?")
            st.markdown(f"""
            <div style='font-size:11px;color:#1A7C44;background:#EDFAF2;border:1px solid #B2EAC8;border-radius:8px;padding:8px 12px;font-family:DM Mono,monospace'>
            ✅ Order: {order_str} &nbsp;|&nbsp; Train: {n_train} minggu &nbsp;|&nbsp; AIC: {aic_val}
            </div>""", unsafe_allow_html=True)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
FC_START    = "2026-01-01"
fc_df       = pd.DataFrame()
using_real  = False
model_mape  = None

if SARIMA_OK and model_trained:
    try:
        raw_fc = sarima_forecast(sel_villa, horizon=fc_horizon_weeks, target_end_date="2026-06-30")
        if raw_fc is not None and not raw_fc.empty and "error" not in raw_fc.columns:
            if "predicted_occupancy" in raw_fc.columns:
                raw_fc["predicted"]  = (raw_fc["predicted_occupancy"] * 100).round(1)
                raw_fc["lower"]      = (raw_fc["lower_bound"] * 100).round(1)
                raw_fc["upper"]      = (raw_fc["upper_bound"] * 100).round(1)
                raw_fc["month_num"]  = raw_fc.index.month
                raw_fc["month"]      = raw_fc.index.strftime("%b %Y")
                raw_fc               = raw_fc.reset_index()
                fc_df = (
                    raw_fc.groupby(["month","month_num"])
                    .agg(predicted=("predicted","mean"), lower=("lower","min"), upper=("upper","max"))
                    .reset_index()
                )
                fc_df["date"] = pd.to_datetime([f"2026-{m:02d}-01" for m in fc_df["month_num"]])
                fc_df         = fc_df.sort_values("month_num").reset_index(drop=True)
                using_real    = True
                if sarima_get_meta:
                    try: model_mape = sarima_get_meta(sel_villa).get("mape")
                    except Exception: pass
        else:
            err_msg = raw_fc["error"].iloc[0] if raw_fc is not None and "error" in raw_fc.columns else "unknown"
            st.warning(f"Forecast SARIMA gagal: {err_msg}. Menggunakan Seasonal Naive.")
    except Exception as ex:
        st.warning(f"Forecast error: {ex}. Menggunakan Seasonal Naive.")

# ── Seasonal Naive fallback ───────────────────────────────────────────────────
if fc_df.empty:
    vd_naive          = df_occ[df_occ["villa_name"] == sel_villa].copy()
    vd_naive["date"]  = pd.to_datetime(vd_naive["date"])
    vd_naive_weekly   = vd_naive.set_index("date").resample("W")["occupancy_pct"].mean()
    weekly_avg_by_month = vd_naive_weekly.groupby(vd_naive_weekly.index.month).mean().to_dict()
    overall_avg_naive   = vd_naive_weekly.mean() if len(vd_naive_weekly) > 0 else 70.0

    rows_fc_weekly = []
    for d in pd.date_range(FC_START, end="2026-06-30", freq="W"):
        pred = float(np.clip(weekly_avg_by_month.get(d.month, overall_avg_naive), 0, 100))
        ci   = pred * 0.10
        rows_fc_weekly.append({"date": d, "month": d.strftime("%b %Y"), "month_num": d.month,
                                "predicted": round(pred,1), "upper": round(min(100,pred+ci),1),
                                "lower": round(max(0,pred-ci),1)})

    fc_df_weekly = pd.DataFrame(rows_fc_weekly)
    fc_df = (
        fc_df_weekly.groupby(["month","month_num"])
        .agg({"predicted":"mean","lower":"min","upper":"max"})
        .reset_index()
    )
    fc_df["date"] = pd.to_datetime([f"2026-{m:02d}-01" for m in fc_df["month_num"]])
    fc_df         = fc_df.sort_values("month_num").reset_index(drop=True)

if "date" not in fc_df.columns:
    fc_df["date"] = pd.to_datetime([f"2026-{i+1:02d}-01" for i in range(len(fc_df))])
if "month_num" not in fc_df.columns:
    fc_df["month_num"] = pd.to_datetime(fc_df["date"]).dt.month

# ── Historis ──────────────────────────────────────────────────────────────────
vd_hist      = df_occ[df_occ["villa_name"] == sel_villa].copy()
hist_occ_avg = vd_hist["occupancy_pct"].mean() if not vd_hist.empty else 70.0
hist_by_mnum = (
    vd_hist.groupby("month_num")["occupancy_pct"]
    .agg(hist_avg="mean", hist_max="max", hist_min="min")
    .reset_index()
)

# ── Filter aktual 2026 ────────────────────────────────────────────────────────
actual_2026_raw = df_occ[
    (df_occ["villa_name"] == sel_villa) & (df_occ["year"] == 2026)
].copy()

actual_monthly = pd.DataFrame()
if not actual_2026_raw.empty:
    actual_monthly = (
        actual_2026_raw.groupby("month_num")["occupancy_pct"]
        .mean().reset_index()
        .rename(columns={"occupancy_pct": "actual"})
    )
    actual_monthly["actual"] = actual_monthly["actual"].round(1)
    actual_monthly["month"]  = actual_monthly["month_num"].apply(
        lambda m: pd.Timestamp(f"2026-{m:02d}-01").strftime("%b %Y")
    )

has_actual = not actual_monthly.empty

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PREDIKSI VS AKTUAL CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='rule'>", unsafe_allow_html=True)

if has_actual:
    covered = len(actual_monthly)
    st.markdown(f"""
<div class='section-label'>Prediksi vs Aktual Okupansi 2026</div>
<div style='background:#EEF3FF;border:1px solid #C5D7FF;border-radius:10px;
     padding:10px 16px;margin-bottom:14px;font-size:12px;color:#1d4ed8'>
  ✅ <b>Data aktual tersedia</b> — {covered} bulan sudah ter-upload &nbsp;·&nbsp;
  Garis oranye = realisasi &nbsp;·&nbsp; Garis biru = prediksi {'SARIMA' if using_real else 'Seasonal Naive'}
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class='section-label'>Prediksi Okupansi 2026</div>
<div style='background:#FFF8EC;border:1px solid #FFD88A;border-radius:10px;
     padding:10px 16px;margin-bottom:14px;font-size:12px;color:#A05C00'>
  ⏳ <b>Belum ada data aktual 2026</b> — Upload data 2026 lewat halaman Upload
  untuk melihat perbandingan prediksi vs realisasi secara langsung
</div>""", unsafe_allow_html=True)

fig_occ = go.Figure()

if show_ci and not fc_df.empty:
    r_, g_, b_ = safe_hex(vc)
    fig_occ.add_trace(go.Scatter(
        x=list(fc_df["month"]) + list(fc_df["month"][::-1]),
        y=list(fc_df["upper"]) + list(fc_df["lower"][::-1]),
        fill="toself", fillcolor=f"rgba({r_},{g_},{b_},0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="CI 95%", hoverinfo="skip", showlegend=True,
    ))

if not fc_df.empty:
    fig_occ.add_trace(go.Scatter(
        x=fc_df["month"], y=fc_df["predicted"],
        mode="lines+markers",
        name=f"Prediksi {'SARIMA' if using_real else 'Seasonal Naive'}",
        line=dict(color=vc, width=3),
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
    title=dict(
        text=f"{'Prediksi vs Aktual' if has_actual else 'Prediksi'} Okupansi {sel_villa} — 2026",
        font=dict(size=16, color="#111", family="DM Serif Display"),
        x=0, xanchor="left", y=0.97, yanchor="top",
    ),
    legend=dict(orientation="h", y=1.15, x=0, xanchor="left", font_size=12, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, tickangle=-35, tickfont_size=11, linecolor="#E8E8E5",
               type="category", categoryorder="array", categoryarray=list(fc_df["month"])),
    yaxis=dict(showgrid=True, gridcolor="#F0F0EE", ticksuffix="%", range=[0,110], tickfont_size=11),
)
st.plotly_chart(fig_occ, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREDIKSI SUMMARY TABLE PER BULAN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Prediksi Okupansi Per Bulan 2026</div>", unsafe_allow_html=True)

st.markdown(f"""
<div style='background:#F5F7FF;border:1px solid #D8E0FF;border-radius:10px;padding:12px 16px;
     margin-bottom:16px;font-size:12px;color:#3D5BC0'>
  <b>📐 Model:</b> {'SARIMA' if using_real else 'Seasonal Naive'} &nbsp;|&nbsp;
  <b>Vila:</b> {sel_villa} &nbsp;|&nbsp;
  <b>S = 52</b> (tahunan pada data mingguan) &nbsp;|&nbsp;
  <b>Horizon:</b> {fc_horizon_months} bulan ({fc_horizon_weeks} minggu prediksi)
</div>
""", unsafe_allow_html=True)

if not fc_df.empty:
    tbl_df = fc_df.copy()
    if has_actual:
        tbl_df = pd.merge(tbl_df, actual_monthly[["month_num","actual"]], on="month_num", how="left")
    else:
        tbl_df["actual"] = np.nan

    rows_html = ""
    cards_html = ""

    for _, row in tbl_df.iterrows():
        occ_pred      = float(row["predicted"])
        month_label   = row["month"]
        actual_val    = row.get("actual", np.nan)
        has_row_actual = not pd.isna(actual_val)

        oc    = "#1A7C44" if occ_pred >= 80 else "#A05C00" if occ_pred >= 50 else "#C0392B"
        bar_w = int(occ_pred)

        if has_row_actual:
            err_val   = float(actual_val) - occ_pred
            err_color = "#1A7C44" if err_val >= 0 else "#C0392B"
            actual_cell = (
                f"<span style='font-family:DM Mono,monospace;font-size:14px;font-weight:700;color:#f97316'>"
                f"{actual_val:.1f}%</span>"
                f"&nbsp;<span style='font-size:10px;color:{err_color};font-family:DM Mono,monospace'>"
                f"({err_val:+.1f}pp)</span>"
            )
        else:
            actual_cell = "<span style='color:#cbd5e1;font-size:12px;font-family:DM Mono,monospace'>—</span>"

        rows_html += f"""<tr>
          <td style='width:22%'><b style='font-size:15px'>{month_label}</b></td>
          <td style='width:48%'>
            <div style='display:flex;align-items:center;gap:14px'>
              <span style='font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:{oc};min-width:72px'>{occ_pred:.1f}%</span>
              <div style='flex:1'>
                <div style='background:#F5F5F3;border-radius:100px;height:8px;overflow:hidden'>
                  <div style='background:{oc};width:{bar_w}%;height:100%;border-radius:100px'></div>
                </div>
              </div>
            </div>
          </td>
          <td style='width:30%'>{actual_cell}</td>
        </tr>"""

        cards_html += f"""
        <div class='forecast-card'>
          <div class='forecast-card-header'>
            <b style='font-size:16px'>{month_label}</b>
            <div style='text-align:right'>
              <span style='font-family:DM Mono,monospace;font-size:22px;font-weight:700;color:{oc}'>{occ_pred:.1f}%</span>
              {"<br>" + actual_cell if has_row_actual else ""}
            </div>
          </div>
          <div class='forecast-card-body'>
            <div style='background:#F5F5F3;border-radius:100px;height:8px;overflow:hidden'>
              <div style='background:{oc};width:{bar_w}%;height:100%;border-radius:100px'></div>
            </div>
          </div>
        </div>"""

    actual_note = "ter-upload" if has_actual else "belum ada data"
    st.markdown(f"""
    <div class='forecast-table-wrapper'>
      <table class='forecast-table'>
        <thead><tr>
          <th style='width:22%'>Bulan</th>
          <th style='width:48%'>Prediksi Okupansi
            <span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:9px;margin-left:6px'>
              {'SARIMA' if using_real else 'Seasonal Naive'}
            </span>
          </th>
          <th style='width:30%'>Aktual 2026
            <span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:9px;margin-left:6px'>
              {actual_note}
            </span>
          </th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div class='forecast-cards-wrapper'>{cards_html}</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUASI AKURASI AKTUAL vs PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
if has_actual and not fc_df.empty:
    eval_df = pd.merge(
        fc_df[["month_num","month","predicted"]],
        actual_monthly[["month_num","actual","month"]],
        on="month_num", how="inner", suffixes=("_fc","_act"),
    )
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
    st.markdown(
        f"<p style='font-size:12px;color:#888;margin-bottom:16px'>"
        f"Dihitung dari <b>{len(eval_df)} bulan</b> yang sudah tersedia data aktualnya.</p>",
        unsafe_allow_html=True,
    )

    ka, kb, kc, kd = st.columns(4)
    for col, label, val, note, color in [
        (ka, "MAPE Aktual", f"{mape_live:.1f}%", "< 10% Sangat Baik · < 20% Baik",
         "#1A7C44" if mape_live < 10 else "#A05C00" if mape_live < 20 else "#C0392B"),
        (kb, "MAE",  f"{mae_live:.1f}pp",  "Mean Absolute Error (poin persen)",
         "#1A7C44" if mae_live < 5 else "#A05C00" if mae_live < 15 else "#C0392B"),
        (kc, "RMSE", f"{rmse_live:.1f}pp", "Root Mean Squared Error",
         "#1A7C44" if rmse_live < 5 else "#A05C00" if rmse_live < 15 else "#C0392B"),
        (kd, "Bias Model", f"{bias:+.1f}pp", "+ = aktual lebih tinggi (under-forecast)  ·  − = over-forecast",
         "#1d4ed8" if abs(bias) < 3 else "#A05C00"),
    ]:
        with col:
            st.markdown(f"""<div class='exec-metric'>
              <div class='exec-metric-label'>{label}</div>
              <div class='exec-metric-value' style='color:{color};font-size:26px'>{val}</div>
              <div class='exec-metric-delta'>{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    bar_colors = ["#1A7C44" if e >= 0 else "#C0392B" for e in eval_df["error"]]
    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(
        x=eval_df["month"], y=eval_df["error"],
        marker_color=bar_colors, marker_line_width=0,
        customdata=eval_df[["actual","predicted","ape"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>Aktual: %{customdata[0]:.1f}%<br>"
            "Prediksi: %{customdata[1]:.1f}%<br>"
            "Error: %{y:+.1f}pp<br>APE: %{customdata[2]:.1f}%<extra></extra>"
        ),
    ))
    fig_err.add_hline(y=0, line_color="#334155", line_width=1.5)
    for _, row in eval_df.iterrows():
        offset = 2.5 if row["error"] >= 0 else -4.5
        fig_err.add_annotation(
            x=row["month"], y=row["error"] + offset,
            text=f"{row['ape']:.1f}%", showarrow=False,
            font=dict(size=9, color="#334155", family="DM Mono"),
        )
    fig_err.update_layout(
        **LAYOUT, height=300, margin=dict(l=0, r=0, t=60, b=40),
        title=dict(
            text=("Error per Bulan (Aktual − Prediksi)"
                  "  <span style='font-size:11px;color:#94a3b8;font-weight:400'>"
                  "· angka = APE%  ·  hijau = aktual lebih tinggi  ·  merah = aktual lebih rendah</span>"),
            font=dict(size=14, color="#111", family="DM Serif Display"), x=0,
        ),
        xaxis=dict(showgrid=False, tickangle=-35, tickfont_size=11, linecolor="#E8E8E5",
                   type="category", categoryorder="array", categoryarray=list(eval_df["month"])),
        yaxis=dict(showgrid=True, gridcolor="#F0F0EE", ticksuffix="pp", tickfont_size=11, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_err, width='stretch')

    with st.expander("📋 Lihat detail tabel evaluasi per bulan", expanded=False):
        eval_display = eval_df[["month","predicted","actual","error","ape"]].copy()
        eval_display.columns = ["Bulan","Prediksi (%)","Aktual (%)","Error (pp)","APE (%)"]

        def style_error(v):
            if not isinstance(v, (int, float)): return ""
            if v > 0:  return "color: #1A7C44; font-weight: 600"
            if v < 0:  return "color: #C0392B; font-weight: 600"
            return ""

        st.dataframe(
            eval_display.style
                .format({"Prediksi (%)":"{:.1f}","Aktual (%)":"{:.1f}","Error (pp)":"{:+.1f}","APE (%)":"{:.1f}%"})
                .applymap(style_error, subset=["Error (pp)"]),
            width='stretch', hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUASI AKURASI SUMMARY SEMUA VILA
# ══════════════════════════════════════════════════════════════════════════════
trained_villas = []
if SARIMA_OK and sarima_exists:
    for vn in villa_names:
        if sarima_exists(vn):
            trained_villas.append(vn)

if trained_villas:
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Evaluasi Akurasi Model SARIMA — Summary Semua Vila</div>", unsafe_allow_html=True)

    def fmt_metric(v, unit="", decimals=2):
        return f"{v:.{decimals}f}{unit}" if v is not None else "—"

    def mape_color(v):
        if v is None: return "#888"
        if v < 10:    return "#1A7C44"
        if v < 20:    return "#A05C00"
        return "#C0392B"

    def rmse_color(v):
        if v is None: return "#888"
        if v < 5:     return "#1A7C44"
        if v < 15:    return "#A05C00"
        return "#C0392B"

    def fmt_order(raw) -> str:
        import re
        if not raw: return "—"
        raw = str(raw)
        SUBSCRIPT_MAP = {"7":"₇","12":"₁₂","4":"₄","24":"₂₄","52":"₅₂","6":"₆"}
        raw = re.sub(r"\[(\d+)\]$", lambda m: SUBSCRIPT_MAP.get(m.group(1), f"[{m.group(1)}]"), raw)
        raw = re.sub(r"(\d+)$",     lambda m: SUBSCRIPT_MAP.get(m.group(1), m.group(1)), raw)
        return raw

    all_metrics = []
    for vn in villa_names:
        is_trained = vn in trained_villas
        vn_color   = get_color(vn)
        meta = {}
        if is_trained and sarima_get_meta:
            try: meta = sarima_get_meta(vn)
            except Exception: meta = {}

        mape_v    = meta.get("mape") or SARIMA_META_STATIC.get(vn,{}).get("mape")
        rmse_v    = meta.get("rmse") or SARIMA_META_STATIC.get(vn,{}).get("rmse")
        aic_v     = meta.get("aic")  or SARIMA_META_STATIC.get(vn,{}).get("aic")
        order_raw = (f"{meta.get('order',())}{meta.get('seasonal_order','')}"
                     if meta.get("order") and meta.get("seasonal_order")
                     else SARIMA_META_STATIC.get(vn,{}).get("order","—"))
        order_s   = fmt_order(order_raw)
        if aic_v: aic_v = round(aic_v)

        all_metrics.append({"vn":vn,"color":vn_color,"is_trained":is_trained,
                             "mape":mape_v,"rmse":rmse_v,"aic":aic_v,"order":order_s,
                             "mc":mape_color(mape_v),"rc":rmse_color(rmse_v)})

    rows_acc = ""
    for m in all_metrics:
        vn       = m["vn"]
        area     = VILLA_INSIGHTS.get(vn,{}).get("area","")
        vn_color = m["color"]
        dot      = f"<span class='villa-dot' style='background:{vn_color}'></span>"
        order    = m["order"] or "—"

        if not m["is_trained"]:
            rows_acc += f"""<tr style='opacity:0.45'>
              <td>{dot}<span style='font-weight:600'>{vn}</span>
                <br><span style='font-size:10px;color:#aaa;font-family:DM Mono,monospace'>{area}</span></td>
              <td colspan='6' style='font-size:11px;color:#bbb;font-family:DM Mono,monospace;text-align:center'>
                ⏳ Belum dilatih — Train model untuk melihat metrik</td>
            </tr>"""
        else:
            if ")(" in order:
                parts       = order.split(")(")
                arima_part  = parts[0] + ")"
                seas_part   = "(" + parts[1]
                order_html  = (f"<span style='font-size:13px;font-weight:700;font-family:DM Mono,monospace;color:#7C3AED'>{arima_part}</span>"
                               f"<span style='font-size:12px;font-weight:600;font-family:DM Mono,monospace;color:#A855F7'>{seas_part}</span>")
            else:
                order_html = f"<span style='font-size:13px;font-weight:700;font-family:DM Mono,monospace;color:#7C3AED'>{order}</span>"

            rows_acc += f"""<tr>
              <td>{dot}<b>{vn}</b><br><span style='font-size:10px;color:#aaa;font-family:DM Mono,monospace'>{area}</span></td>
              <td><span style='font-size:17px;font-weight:700;font-family:DM Mono,monospace;color:{m["mc"]}'>{fmt_metric(m["mape"],"%",2)}</span></td>
              <td><span style='font-size:15px;font-weight:700;font-family:DM Mono,monospace;color:{m["rc"]}'>{fmt_metric(m["rmse"],"",2)}</span></td>
              <td><span style='font-size:13px;font-family:DM Mono,monospace;color:#3D6BE8'>{fmt_metric(m["aic"],"",0)}</span></td>
              <td>{order_html}<br><span style='font-size:9px;color:#94a3b8;font-family:DM Mono,monospace'>S=52 (tahunan)</span></td>
            </tr>"""

    st.markdown(f"""
    <table class='summary-acc-table'>
      <thead><tr>
        <th>Vila</th>
        <th>MAPE<br><span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:8px'>Error rata-rata %</span></th>
        <th>RMSE<br><span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:8px'>Error kuadrat</span></th>
        <th>AIC<br><span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:8px'>Fit model</span></th>
        <th>Model SARIMA Terpilih<br><span style='font-weight:400;text-transform:none;letter-spacing:0;font-size:8px'>(p,d,q)(P,D,Q)[S]</span></th>
      </tr></thead>
      <tbody>{rows_acc}</tbody>
    </table>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ANALISIS DESKRIPTIF ADR & OKUPANSI
# ══════════════════════════════════════════════════════════════════════════════
def build_monthly_data_filtered(villa_name: str, year_filter=None):
    occ_raw           = df_occ[df_occ["villa_name"] == villa_name].copy()
    occ_raw["date"]   = pd.to_datetime(occ_raw["date"])
    if year_filter:
        occ_raw = occ_raw[occ_raw["year"].isin(year_filter)]
    occ_raw["period"] = occ_raw["date"].dt.to_period("M").astype(str)
    monthly_occ = (
        occ_raw.groupby("period")["occupancy_pct"].mean().reset_index()
        .rename(columns={"occupancy_pct": "occ"})
    )
    corr_info = ADR_STATIC.get(villa_name, {}).copy()
    corr_info["live"] = False
    corr_info.setdefault("r", 0)

    if df_fin is None or df_fin.empty or "avg_daily_revenue" not in df_fin.columns:
        monthly_occ["adr_m"] = np.nan
        return monthly_occ, corr_info, pd.DataFrame()

    fin_raw           = df_fin[df_fin["villa_name"] == villa_name].copy()
    fin_raw["date"]   = pd.to_datetime(fin_raw["date"])
    if year_filter:
        fin_raw = fin_raw[fin_raw["year"].isin(year_filter)]
    fin_filled        = (fin_raw[fin_raw["for_modeling"] == 1].copy()
                         if "for_modeling" in fin_raw.columns
                         else fin_raw[fin_raw["avg_daily_revenue"] > 0].copy())
    if fin_filled.empty:
        monthly_occ["adr_m"] = np.nan
        return monthly_occ, corr_info, pd.DataFrame()

    fin_filled["period"] = fin_filled["date"].dt.to_period("M").astype(str)
    monthly_adr = (
        fin_filled.groupby("period")["avg_daily_revenue"].mean().reset_index()
        .rename(columns={"avg_daily_revenue": "adr_m"})
    )
    monthly_adr["adr_m"] = monthly_adr["adr_m"] / 1_000_000
    monthly = pd.merge(monthly_occ, monthly_adr, on="period", how="left").sort_values("period").reset_index(drop=True)

    # Build daily merged for scatter
    occ_daily = occ_raw[["date","occupancy_pct"]].copy()
    daily_merged = pd.merge(occ_daily, fin_filled[["date","avg_daily_revenue"]], on="date", how="inner").dropna()

    try:
        if len(daily_merged) >= 10:
            r_val, _ = stats.pearsonr(daily_merged["occupancy_pct"], daily_merged["avg_daily_revenue"])
            corr_info = {
                "r": round(r_val,2), "n": len(daily_merged), "live": True,
                "adr_median": round(fin_filled["avg_daily_revenue"].median()/1_000_000,2),
                "adr_mean":   round(fin_filled["avg_daily_revenue"].mean()/1_000_000,2),
                "adr_max":    round(fin_filled["avg_daily_revenue"].max()/1_000_000,2),
                "adr_min":    round(fin_filled["avg_daily_revenue"].min()/1_000_000,2),
                "adr_std":    round(fin_filled["avg_daily_revenue"].std()/1_000_000,2),
                "elastisitas": ADR_STATIC.get(villa_name,{}).get("elastisitas","Semi-Elastis"),
                "area":        ADR_STATIC.get(villa_name,{}).get("area",""),
                "strategy":    ADR_STATIC.get(villa_name,{}).get("strategy","dynamic_moderate"),
            }
    except Exception:
        pass

    return monthly, corr_info, daily_merged


st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("""
<div class='anim'>
  <div class='section-label'>📈 Analisis Deskriptif — Keterhubungan ADR & Okupansi</div>
  <p style='font-size:13px;color:#666;margin-bottom:4px'>
    Rata-rata okupansi bulanan (semua hari) vs rata-rata ADR bulanan (hanya hari ada revenue).
    Konsisten dengan metodologi notebook penelitian.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Filters for descriptive section ──────────────────────────────────────────
all_areas   = sorted(set(VILLA_INSIGHTS.get(v,{}).get("area","") for v in villa_names if VILLA_INSIGHTS.get(v,{}).get("area","")))
all_years   = sorted(df_occ["year"].unique().tolist())

fa, fb, fc_col = st.columns([1.5, 2.5, 2])
with fa:
    sel_area_desc = st.multiselect(
        "Filter Area", options=all_areas, default=all_areas, key="desc_area",
        placeholder="Semua area"
    )
with fb:
    filtered_villa_names = [v for v in villa_names
                            if VILLA_INSIGHTS.get(v,{}).get("area","") in (sel_area_desc or all_areas)]
    sel_villas_desc = st.multiselect(
        "Filter Vila", options=filtered_villa_names, default=filtered_villa_names, key="desc_villa",
        placeholder="Semua vila"
    )
with fc_col:
    sel_years_desc = st.multiselect(
        "Filter Tahun Historis", options=all_years, default=all_years, key="desc_year",
        placeholder="Semua tahun"
    )

display_villas = sel_villas_desc if sel_villas_desc else filtered_villa_names
year_filter    = sel_years_desc if sel_years_desc else None

st.markdown("""
<div style='font-family:DM Mono,monospace;font-size:10px;letter-spacing:.16em;
            text-transform:uppercase;color:#3D6BE8;margin:20px 0 12px;
            padding-left:10px;border-left:3px solid #3D6BE8'>
  Tren Musiman ADR vs Okupansi per Vila (Dual-Axis)
</div>""", unsafe_allow_html=True)

st.markdown("""
<p style='font-size:12px;color:#888;margin-bottom:16px'>
  Batang = rata-rata okupansi bulanan (sumbu kanan, %). &nbsp;
  Garis oranye = rata-rata ADR bulanan (sumbu kiri, Juta Rupiah). &nbsp;
  ADR hanya dihitung dari hari vila terisi dengan revenue &gt; 0.
</p>""", unsafe_allow_html=True)

if not display_villas:
    st.info("Pilih minimal satu vila untuk menampilkan analisis.")
else:
    tab_labels   = [v.replace(" Villas","") for v in display_villas]
    tab_villas_d = st.tabs(tab_labels)

    for tab_idx, (tab, vn) in enumerate(zip(tab_villas_d, display_villas)):
        with tab:
            vc4              = get_color(vn)
            r_hex, g_hex, b_hex = safe_hex(vc4)
            monthly, cd2, daily_merged = build_monthly_data_filtered(vn, year_filter=year_filter)

            if monthly.empty:
                st.info(f"Belum ada data ocupansi untuk {vn}.")
                continue

            has_adr = "adr_m" in monthly.columns and monthly["adr_m"].notna().any()
            if not has_adr:
                monthly["adr_m"] = np.nan

            # ── Dual-axis trend chart ─────────────────────────────────────────
            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            fig_dual.add_trace(go.Bar(
                x=monthly["period"], y=monthly["occ"], name="Okupansi (%)",
                marker_color=f"rgba({r_hex},{g_hex},{b_hex},0.55)",
                marker_line_color=f"rgba({r_hex},{g_hex},{b_hex},0.85)", marker_line_width=1.2,
                hovertemplate="<b>%{x}</b><br>Ocupansi: %{y:.1f}%<extra></extra>",
            ), secondary_y=True)

            adr_data = monthly.dropna(subset=["adr_m"])
            if not adr_data.empty:
                fig_dual.add_trace(go.Scatter(
                    x=adr_data["period"], y=adr_data["adr_m"], name="ADR (Juta Rupiah)",
                    mode="lines+markers", line=dict(color="#f97316", width=2.5),
                    marker=dict(size=7, color="#f97316", line=dict(color="#F7F7F5", width=1.5)),
                    hovertemplate="<b>%{x}</b><br>ADR: Rp%{y:.2f} Juta<extra></extra>",
                ), secondary_y=False)

            for period_val in monthly["period"]:
                try:
                    mn_num = int(str(period_val)[-2:])
                    if mn_num in PEAK_MONTHS:
                        fig_dual.add_vrect(x0=period_val, x1=period_val,
                                           fillcolor="rgba(255,200,0,0.07)", layer="below", line_width=0)
                except Exception:
                    pass

            r_val    = cd2.get("r", 0)
            n_obs    = cd2.get("n", "—")
            data_src = "live" if cd2.get("live") else "statis"

            fig_dual.update_layout(
                **LAYOUT, height=360, margin=dict(l=0, r=0, t=60, b=40),
                title=dict(
                    text=(f"Tren Musiman ADR & Okupansi — {vn}"
                          f"&nbsp;&nbsp;<span style='font-size:13px;color:#666'>"
                          f"Korelasi Pearson: <b>r = {r_val:+.2f}</b>"
                          f"&nbsp;·&nbsp;<span style='font-size:11px;color:#aaa'>n={n_obs} hari · {data_src}</span></span>"),
                    font=dict(size=14, color="#111", family="DM Serif Display"), x=0, xanchor="left",
                ),
                legend=dict(orientation="h", y=1.12, x=0, xanchor="left", font_size=11, bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(showgrid=False, tickangle=-45, tickfont_size=10, linecolor="#E8E8E5", type="category"),
                bargap=0.25,
            )
            fig_dual.update_yaxes(title_text="ADR (Juta Rupiah)", secondary_y=False,
                                   showgrid=True, gridcolor="#F0F0EE", ticksuffix=" Jt", tickfont_size=11, rangemode="tozero")
            fig_dual.update_yaxes(title_text="Ocupansi (%)", secondary_y=True,
                                   showgrid=False, ticksuffix="%", range=[0,130], tickfont_size=11)
            st.plotly_chart(fig_dual, width='stretch')

            # ── Korelasi scatter: ADR vs Okupansi ────────────────────────────
            if not daily_merged.empty and len(daily_merged) >= 5:
                st.markdown(f"""
                <div style='font-family:DM Mono,monospace;font-size:10px;letter-spacing:.14em;
                            text-transform:uppercase;color:#7c3aed;margin:18px 0 10px;
                            padding-left:10px;border-left:3px solid #7c3aed'>
                  Korelasi Harga (ADR) vs Okupansi — {vn}
                </div>""", unsafe_allow_html=True)

                scatter_x = daily_merged["occupancy_pct"].values
                scatter_y = daily_merged["avg_daily_revenue"].values / 1_000_000

                # Regression line
                try:
                    m_coef, b_coef = np.polyfit(scatter_x, scatter_y, 1)
                    x_line = np.linspace(scatter_x.min(), scatter_x.max(), 100)
                    y_line = m_coef * x_line + b_coef
                    has_reg = True
                except Exception:
                    has_reg = False

                r_val_sc = cd2.get("r", 0)
                r2_val   = round(r_val_sc ** 2, 3)

                # Color points by ADR level
                adr_med_sc = np.median(scatter_y)
                point_colors = [f"rgba({r_hex},{g_hex},{b_hex},0.65)" if y >= adr_med_sc
                                else f"rgba({r_hex},{g_hex},{b_hex},0.30)" for y in scatter_y]

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=scatter_x, y=scatter_y,
                    mode="markers",
                    marker=dict(size=6, color=point_colors, line=dict(color=vc4, width=0.5)),
                    name="Hari Observasi",
                    hovertemplate="Ocupansi: %{x:.1f}%<br>ADR: Rp%{y:.2f} Juta<extra></extra>",
                ))

                if has_reg:
                    fig_scatter.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode="lines",
                        line=dict(color="#f97316", width=2.5, dash="dash"),
                        name=f"Regresi linier (r={r_val_sc:+.2f})",
                        hoverinfo="skip",
                    ))

                # Add r annotation
                fig_scatter.add_annotation(
                    xref="paper", yref="paper", x=0.97, y=0.97,
                    text=f"<b>r = {r_val_sc:+.2f}</b>  |  R² = {r2_val:.3f}  |  n = {len(scatter_x)}",
                    showarrow=False, bgcolor="#fff", bordercolor=vc4, borderwidth=1,
                    font=dict(size=12, color=vc4, family="DM Mono"),
                    align="right",
                )

                fig_scatter.update_layout(
                    **LAYOUT, height=320, margin=dict(l=0, r=0, t=50, b=40),
                    title=dict(
                        text=f"Scatter: ADR vs Tingkat Okupansi — {vn}",
                        font=dict(size=13, color="#111", family="DM Serif Display"), x=0,
                    ),
                    legend=dict(orientation="h", y=1.12, x=0, font_size=11, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(title="Ocupansi (%)", showgrid=True, gridcolor="#F0F0EE",
                               ticksuffix="%", tickfont_size=11),
                    yaxis=dict(title="ADR (Juta Rupiah)", showgrid=True, gridcolor="#F0F0EE",
                               ticksuffix=" Jt", tickfont_size=11, rangemode="tozero"),
                )
                st.plotly_chart(fig_scatter, width='stretch')

                # Correlation interpretation card
                abs_r = abs(r_val_sc)
                if abs_r >= 0.7:
                    corr_strength = "Kuat"
                    corr_color    = "#1A7C44"
                    corr_bg       = "#EDFAF2"
                    corr_border   = "#B2EAC8"
                    corr_icon     = "🔗"
                elif abs_r >= 0.4:
                    corr_strength = "Moderat"
                    corr_color    = "#2563EB"
                    corr_bg       = "#EEF3FF"
                    corr_border   = "#C5D7FF"
                    corr_icon     = "〰️"
                elif abs_r >= 0.2:
                    corr_strength = "Lemah"
                    corr_color    = "#A05C00"
                    corr_bg       = "#FFF8EC"
                    corr_border   = "#FFD88A"
                    corr_icon     = "📉"
                else:
                    corr_strength = "Sangat Lemah / Tidak Ada"
                    corr_color    = "#64748b"
                    corr_bg       = "#f1f5f9"
                    corr_border   = "#e2e8f0"
                    corr_icon     = "➖"

                direction = "positif — harga naik seiring ocupansi naik" if r_val_sc >= 0 else "negatif — harga turun saat ocupansi naik"

                adr_med_disp = cd2.get("adr_median", ADR_STATIC.get(vn,{}).get("median_adr", 0))
                adr_hi_disp  = cd2.get("adr_max",    ADR_STATIC.get(vn,{}).get("adr_max", 0))
                adr_lo_disp  = cd2.get("adr_min",    ADR_STATIC.get(vn,{}).get("adr_min", 0))

                st.markdown(f"""
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px'>
                  <div style='background:{corr_bg};border:1px solid {corr_border};border-radius:10px;padding:14px;border-top:3px solid {corr_color}'>
                    <div style='font-family:DM Mono,monospace;font-size:9px;color:{corr_color};text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>
                      {corr_icon} Korelasi Pearson
                    </div>
                    <div style='font-size:26px;font-weight:800;font-family:DM Mono,monospace;color:{corr_color};margin-bottom:4px'>
                      r = {r_val_sc:+.2f}
                    </div>
                    <div style='font-size:12px;font-weight:700;color:{corr_color};margin-bottom:4px'>{corr_strength}</div>
                    <div style='font-size:11px;color:#555;line-height:1.5'>
                      Arah: {direction}<br>
                      R² = {r2_val:.3f} ({r2_val*100:.1f}% variasi ADR dijelaskan oleh ocupansi)<br>
                      Observasi: {len(scatter_x)} hari
                    </div>
                  </div>
                  <div style='background:#fff;border:1px solid #E8E8E5;border-radius:10px;padding:14px;border-top:3px solid {vc4}'>
                    <div style='font-family:DM Mono,monospace;font-size:9px;color:#aaa;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>Profil Harga</div>
                    <div style='font-size:22px;font-weight:700;font-family:DM Mono,monospace;color:{vc4};margin-bottom:4px'>
                      Rp{adr_med_disp:.2f} Juta
                    </div>
                    <div style='font-size:11px;color:#888'>Median ADR (hari ada revenue)</div>
                    <div style='font-size:11px;color:#aaa;margin-top:4px'>
                      Range: Rp{adr_lo_disp:.2f} Juta – Rp{adr_hi_disp:.2f} Juta
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Data keuangan tidak cukup untuk analisis korelasi.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:28px 0 8px;color:#ccc;font-size:11px;
    font-family:DM Mono,monospace;letter-spacing:.08em'>
  VILLAS R US ANALYTICS · STRATEGI OKUPANSI · SARIMA FORECAST · 2026
</div>""", unsafe_allow_html=True)