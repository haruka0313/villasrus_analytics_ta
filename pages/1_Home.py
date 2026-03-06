import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings

warnings.filterwarnings("ignore")

from utils.auth import get_cookie_manager, set_session, load_from_cookie, logout
from utils.sidebar import render_sidebar

# ─── COOKIES ─────────────────────────────────────────────────────────────────
cookies = get_cookie_manager()  # kalau belum ready, st.stop() otomatis di dalam fungsi

# ─── AUTH CHECK ──────────────────────────────────────────────────────────────
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

# ─── RENDER SIDEBAR ──────────────────────────────────────────────────────────
render_sidebar(cookies)

from database import get_occupancy_data, get_financial_data, get_villas

# ── Static Config (sama persis dengan sarima_forecast.py) ────────────────────
VILLA_INSIGHTS = {
    "Briana Villas":  {"elastisitas": "Semi-Elastis", "color": "#3D6BE8", "area": "Canggu"},
    "Castello Villas":{"elastisitas": "Semi-Elastis", "color": "#7c3aed", "area": "Canggu"},
    "Elina Villas":   {"elastisitas": "Semi-Elastis", "color": "#059669", "area": "Canggu"},
    "Isola Villas":   {"elastisitas": "Elastis",      "color": "#db2777", "area": "Canggu"},
    "Eindra Villas":  {"elastisitas": "Inelastis",    "color": "#d97706", "area": "Seminyak"},
    "Esha Villas":    {"elastisitas": "Semi-Elastis", "color": "#b45309", "area": "Seminyak"},
    "Ozamiz Villas":  {"elastisitas": "Inelastis",    "color": "#9333ea", "area": "Seminyak"},
}

SARIMA_META_STATIC = {
    "Briana Villas":  {"order": "(2,1,0)(0,1,0)₅₂", "mape": 51.96, "rmse": 0.39, "aic": 88},
    "Castello Villas":{"order": "(0,1,1)(1,1,0)₅₂", "mape": 45.07, "rmse": 0.41, "aic": 16},
    "Elina Villas":   {"order": "(0,0,1)(0,1,1)₅₂", "mape": 120.40,"rmse": 0.47, "aic": 26},
    "Isola Villas":   {"order": "(2,1,0)(0,1,0)₅₂", "mape": 54.86, "rmse": 0.40, "aic": 88},
    "Eindra Villas":  {"order": "(0,0,0)(1,1,0)₅₂", "mape": 31.95, "rmse": 0.30, "aic": 24},
    "Esha Villas":    {"order": "(1,0,1)(1,1,0)₅₂", "mape": 44.98, "rmse": 0.34, "aic": 25},
    "Ozamiz Villas":  {"order": "(1,0,0)(1,1,0)₅₂", "mape": 30.92, "rmse": 0.30, "aic": 14},
}

# Proyeksi Semester I 2026 per vila (dari Tabel 4.10 BAB IV)
FORECAST_2026 = {
    "Briana Villas":  {"Jan":76.2,"Feb":75.0,"Mar":51.4,"Apr":75.0,"May":88.6,"Jun":53.6,"avg":69.97},
    "Castello Villas":{"Jan":74.4,"Feb":77.3,"Mar":72.1,"Apr":87.1,"May":93.6,"Jun":99.1,"avg":83.94},
    "Elina Villas":   {"Jan":22.2,"Feb":0.0, "Mar":12.9,"Apr":16.1,"May":0.0, "Jun":32.0,"avg":13.87},
    "Isola Villas":   {"Jan":76.2,"Feb":75.0,"Mar":51.4,"Apr":75.0,"May":88.6,"Jun":53.6,"avg":69.97},
    "Eindra Villas":  {"Jan":78.6,"Feb":60.7,"Mar":80.0,"Apr":64.3,"May":61.4,"Jun":73.2,"avg":69.70},
    "Esha Villas":    {"Jan":77.9,"Feb":49.7,"Mar":68.9,"Apr":64.8,"May":55.0,"Jun":76.3,"avg":65.43},
    "Ozamiz Villas":  {"Jan":93.3,"Feb":74.1,"Mar":72.1,"Apr":64.3,"May":43.3,"Jun":90.4,"avg":72.92},
}

MONTHS_ORDER = ["Jan","Feb","Mar","Apr","May","Jun"]
PEAK_MONTHS  = [6, 7, 8, 12, 1]
AREA_ORDER   = ["Canggu", "Seminyak"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;900&display=swap');

  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background: #f0f6ff !important; color: #0f172a; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stSidebarNav"] { display: none !important; }
  [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }

  /* ── HERO ── */
  .hero-wrap {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0f4c81 100%);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }
  .hero-wrap::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-wrap::after {
    content: '';
    position: absolute; bottom: -80px; left: 30%;
    width: 400px; height: 200px;
    background: radial-gradient(ellipse, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: .22em;
    text-transform: uppercase; color: #38bdf8;
    margin-bottom: 10px;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 38px; font-weight: 900;
    color: #f8fafc; line-height: 1.15;
    margin-bottom: 12px;
  }
  .hero-title em { color: #38bdf8; font-style: italic; }
  .hero-sub { font-size: 14px; color: #94a3b8; max-width: 520px; line-height: 1.65; }
  .hero-badges { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
  .hero-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px; padding: 5px 14px;
    font-size: 11px; color: #cbd5e1;
    font-family: 'DM Mono', monospace;
  }
  .hero-badge strong { color: #38bdf8; }

  /* ── KPI CARDS ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    position: relative; overflow: hidden;
    box-shadow: 0 2px 12px rgba(15,23,42,0.06);
    transition: transform .2s, box-shadow .2s;
  }
  .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(15,23,42,0.1); }
  .kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, #3D6BE8);
    border-radius: 16px 16px 0 0;
  }
  .kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: .16em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'DM Mono', monospace;
    font-size: 32px; font-weight: 700;
    color: #0f172a; line-height: 1; margin-bottom: 6px;
  }
  .kpi-delta { font-size: 12px; color: #64748b; }
  .kpi-delta.up   { color: #16a34a; }
  .kpi-delta.down { color: #dc2626; }
  .kpi-icon {
    position: absolute; top: 18px; right: 18px;
    font-size: 22px; opacity: 0.15;
  }

  /* ── SECTION LABEL ── */
  .section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: .16em;
    text-transform: uppercase; color: #64748b;
    padding-bottom: 8px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 18px; margin-top: 8px;
  }
  .section-label span { color: #3D6BE8; }

  /* ── AREA BADGE ── */
  .area-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px; margin-top: 20px;
  }
  .area-pill {
    font-family: 'DM Mono', monospace;
    font-size: 10px; font-weight: 600;
    padding: 4px 14px; border-radius: 100px;
    letter-spacing: .1em; text-transform: uppercase;
  }
  .area-pill.canggu   { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
  .area-pill.seminyak { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
  .area-divider { flex: 1; height: 1px; background: #e2e8f0; }

  /* ── VILLA CARD ── */
  .villa-row {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: grid;
    grid-template-columns: 200px 1fr 110px 110px 110px 160px;
    align-items: center; gap: 16px;
    transition: box-shadow .2s;
    border-left: 4px solid var(--vc);
  }
  .villa-row:hover { box-shadow: 0 4px 16px rgba(15,23,42,0.08); }
  .villa-name { font-size: 14px; font-weight: 700; color: #0f172a; }
  .villa-area { font-size: 10px; color: #94a3b8; font-family: 'DM Mono', monospace; text-transform: uppercase; margin-top: 2px; }
  .villa-occ-big {
    font-family: 'DM Mono', monospace;
    font-size: 22px; font-weight: 700;
  }
  .villa-stat-label { font-size: 9px; color: #94a3b8; font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 3px; }
  .villa-stat-val   { font-family: 'DM Mono', monospace; font-size: 14px; font-weight: 600; color: #334155; }
  .mini-bar-wrap { background: #f1f5f9; border-radius: 100px; height: 6px; overflow: hidden; margin-top: 4px; }
  .mini-bar      { height: 100%; border-radius: 100px; }

  /* ── FORECAST HEATMAP ROW ── */
  .heatmap-table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 14px; overflow: hidden; border: 1px solid #e2e8f0; }
  .heatmap-table th {
    background: #0f172a; color: #94a3b8;
    font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: .14em;
    text-transform: uppercase; padding: 12px 14px; text-align: left;
  }
  .heatmap-table td { padding: 11px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  .heatmap-table tr:last-child td { border-bottom: none; }
  .heatmap-table tr:hover td { background: #f8fafc; }
  .heat-cell {
    width: 54px; height: 32px;
    border-radius: 6px;
    display: inline-flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 600;
  }

  /* ── ALERT CARDS ── */
  .alert-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 24px; }
  .alert-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 16px 18px;
    border-left: 4px solid var(--alert-color);
  }
  .alert-title { font-size: 12px; font-weight: 700; color: #0f172a; margin-bottom: 6px; }
  .alert-body  { font-size: 12px; color: #64748b; line-height: 1.6; }
  .alert-body b { color: #0f172a; }

  /* ── INSIGHT CHIP ── */
  .insight-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .insight-chip {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 100px; padding: 5px 14px;
    font-size: 11px; color: #475569;
    font-family: 'DM Mono', monospace;
  }
  .insight-chip.green { background: #f0fdf4; border-color: #bbf7d0; color: #15803d; }
  .insight-chip.red   { background: #fef2f2; border-color: #fecaca; color: #b91c1c; }
  .insight-chip.blue  { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
  .insight-chip.amber { background: #fffbeb; border-color: #fde68a; color: #92400e; }

  .rule { border: none; border-top: 1px solid #e2e8f0; margin: 24px 0; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .anim-1 { animation: fadeUp .4s ease .05s both; }
  .anim-2 { animation: fadeUp .4s ease .15s both; }
  .anim-3 { animation: fadeUp .4s ease .25s both; }
  .anim-4 { animation: fadeUp .4s ease .35s both; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    return get_occupancy_data(), get_financial_data(), get_villas()

df_occ, df_fin, df_villas = load_data()

VILLA_COLORS = {}
if df_villas is not None and not df_villas.empty:
    VILLA_COLORS = {r["villa_name"]: r["color_hex"] for _, r in df_villas.iterrows()}

def get_color(name):
    return VILLA_COLORS.get(name, VILLA_INSIGHTS.get(name, {}).get("color", "#3D6BE8"))

def safe_hex(hex_color):
    h = hex_color.lstrip("#")
    if len(h) != 6: h = "3D6BE8"
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def occ_color(v):
    if v >= 80: return "#16a34a"
    if v >= 60: return "#d97706"
    if v >= 40: return "#ea580c"
    return "#dc2626"

def heat_bg(v):
    """Return (bg, text) color tuple based on occupancy value."""
    if v >= 85: return "#166534", "#dcfce7"
    if v >= 70: return "#15803d", "#f0fdf4"
    if v >= 55: return "#ca8a04", "#fefce8"
    if v >= 40: return "#ea580c", "#fff7ed"
    if v >= 20: return "#dc2626", "#fef2f2"
    return "#991b1b", "#fce7f3"

# ── Pre-compute metrics ───────────────────────────────────────────────────────
villa_names = list(VILLA_INSIGHTS.keys())

hist_metrics = {}
if df_occ is not None and not df_occ.empty:
    df_occ = df_occ.copy()
    df_occ["date"]  = pd.to_datetime(df_occ["date"])
    df_occ["year"]  = df_occ["date"].dt.year
    df_occ["month_num"] = df_occ["date"].dt.month

    for vn in villa_names:
        vd = df_occ[df_occ["villa_name"] == vn]
        if vd.empty:
            hist_metrics[vn] = {"mean": 0, "peak_mean": 0, "low_mean": 0, "trend": 0}
            continue
        mean_all  = vd["occupancy_pct"].mean()
        peak_d    = vd[vd["month_num"].isin(PEAK_MONTHS)]
        off_d     = vd[~vd["month_num"].isin(PEAK_MONTHS)]
        # yoy trend: compare last year vs year before
        years = sorted(vd["year"].unique())
        trend = 0.0
        if len(years) >= 2:
            y_last = vd[vd["year"] == years[-1]]["occupancy_pct"].mean()
            y_prev = vd[vd["year"] == years[-2]]["occupancy_pct"].mean()
            trend = y_last - y_prev
        hist_metrics[vn] = {
            "mean": round(mean_all, 1),
            "peak_mean": round(peak_d["occupancy_pct"].mean(), 1) if not peak_d.empty else 0,
            "low_mean":  round(off_d["occupancy_pct"].mean(), 1)  if not off_d.empty else 0,
            "trend": round(trend, 1),
        }
else:
    for vn in villa_names:
        hist_metrics[vn] = {"mean": 0, "peak_mean": 0, "low_mean": 0, "trend": 0}

# Portfolio-level KPIs
all_avgs   = [FORECAST_2026[vn]["avg"] for vn in villa_names]
port_avg   = round(np.mean(all_avgs), 1)
best_villa = max(FORECAST_2026, key=lambda v: FORECAST_2026[v]["avg"])
risk_villas= [vn for vn in villa_names if FORECAST_2026[vn]["avg"] < 50]

canggu_villas   = [vn for vn in villa_names if VILLA_INSIGHTS[vn]["area"] == "Canggu"]
seminyak_villas = [vn for vn in villa_names if VILLA_INSIGHTS[vn]["area"] == "Seminyak"]
canggu_avg   = round(np.mean([FORECAST_2026[v]["avg"] for v in canggu_villas]),1)
seminyak_avg = round(np.mean([FORECAST_2026[v]["avg"] for v in seminyak_villas]),1)

# Peak month across portfolio
month_avgs = {m: np.mean([FORECAST_2026[v][m] for v in villa_names]) for m in MONTHS_ORDER}
best_month = max(month_avgs, key=month_avgs.get)

# ══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════════════
import datetime
hour = datetime.datetime.now().hour
greeting = "Selamat Pagi" if hour < 12 else "Selamat Siang" if hour < 17 else "Selamat Malam"
username = st.session_state.get("username", "Administrator")

st.markdown(f"""
<div class='hero-wrap anim-1'>
  <div class='hero-eyebrow'>Villas R Us · Analytics Platform · Dashboard Utama</div>
  <div class='hero-title'>{greeting}, <em>{username}</em>.</div>
  <div class='hero-sub'>
    Ringkasan performa seluruh <strong style='color:#e2e8f0'>7 unit vila</strong>
    di kawasan <strong style='color:#e2e8f0'>Canggu</strong> dan
    <strong style='color:#e2e8f0'>Seminyak</strong> —
    berdasarkan proyeksi model SARIMA Semester I 2026 dan data historis 2023–2025.
  </div>
  <div class='hero-badges'>
    <span class='hero-badge'>📍 <strong>Canggu</strong> · 4 Vila</span>
    <span class='hero-badge'>📍 <strong>Seminyak</strong> · 3 Vila</span>
    <span class='hero-badge'>🤖 <strong>SARIMA</strong> S=52 · m=weekly</span>
    <span class='hero-badge'>📅 Proyeksi <strong>Jan–Jun 2026</strong></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
delta_cs = seminyak_avg - canggu_avg
dc_sign  = "▲" if delta_cs >= 0 else "▼"

st.markdown(f"""
<div class='kpi-grid anim-2'>
  <div class='kpi-card' style='--accent:#3D6BE8'>
    <div class='kpi-icon'>🏝</div>
    <div class='kpi-label'>Portfolio Avg · Sem I 2026</div>
    <div class='kpi-value'>{port_avg}%</div>
    <div class='kpi-delta'>proyeksi rata-rata 7 vila</div>
  </div>
  <div class='kpi-card' style='--accent:#16a34a'>
    <div class='kpi-icon'>🏆</div>
    <div class='kpi-label'>Top Performer</div>
    <div class='kpi-value' style='font-size:20px;padding-top:4px'>{best_villa.replace(" Villas","")}</div>
    <div class='kpi-delta up'>▲ {FORECAST_2026[best_villa]["avg"]}% avg hunian</div>
  </div>
  <div class='kpi-card' style='--accent:#f97316'>
    <div class='kpi-icon'>📅</div>
    <div class='kpi-label'>Peak Month (Portfolio)</div>
    <div class='kpi-value'>{best_month} '26</div>
    <div class='kpi-delta'>rata-rata {month_avgs[best_month]:.1f}% seluruh vila</div>
  </div>
  <div class='kpi-card' style='--accent:#dc2626'>
    <div class='kpi-icon'>⚠️</div>
    <div class='kpi-label'>Vila Risiko Tinggi</div>
    <div class='kpi-value'>{len(risk_villas)}/7</div>
    <div class='kpi-delta {"down" if risk_villas else "up"}'>{"· ".join([v.replace(" Villas","") for v in risk_villas]) if risk_villas else "Semua vila di atas 50%"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: PORTFOLIO OVERVIEW — Grouped Bar per Vila per Bulan
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label anim-3'><span>01</span> · Proyeksi Hunian Semester I 2026 — Seluruh Vila</div>", unsafe_allow_html=True)

fig_port = go.Figure()
for vn in villa_names:
    vc = get_color(vn)
    r,g,b = safe_hex(vc)
    vals = [FORECAST_2026[vn][m] for m in MONTHS_ORDER]
    fig_port.add_trace(go.Bar(
        name=vn.replace(" Villas",""),
        x=MONTHS_ORDER, y=vals,
        marker_color=f"rgba({r},{g},{b},0.85)",
        marker_line_color=f"rgba({r},{g},{b},1)",
        marker_line_width=1.2,
        hovertemplate=f"<b>{vn}</b><br>%{{x}} 2026: %{{y:.1f}}%<extra></extra>",
    ))

fig_port.update_layout(
    barmode="group",
    plot_bgcolor="#ffffff", paper_bgcolor="#f0f6ff",
    font_family="Sora", font_color="#334155",
    height=340, margin=dict(l=0,r=0,t=20,b=40),
    legend=dict(orientation="h", y=1.08, x=0, xanchor="left", font_size=11,
                bgcolor="rgba(0,0,0,0)", itemclick="toggleothers"),
    xaxis=dict(showgrid=False, linecolor="#e2e8f0", tickfont_size=12),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%",
               range=[0,115], tickfont_size=11, title=""),
    bargap=0.18, bargroupgap=0.04,
)
st.plotly_chart(fig_port, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: AREA COMPARISON — Canggu vs Seminyak radar / line
# ══════════════════════════════════════════════════════════════════════════════
col_chart_l, col_chart_r = st.columns([1,1], gap="large")

with col_chart_l:
    st.markdown("<div class='section-label'><span>02</span> · Tren Bulanan — Canggu vs Seminyak</div>", unsafe_allow_html=True)

    canggu_monthly   = [np.mean([FORECAST_2026[v][m] for v in canggu_villas])   for m in MONTHS_ORDER]
    seminyak_monthly = [np.mean([FORECAST_2026[v][m] for v in seminyak_villas]) for m in MONTHS_ORDER]

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=MONTHS_ORDER, y=canggu_monthly, name="Canggu",
        mode="lines+markers",
        line=dict(color="#3D6BE8", width=3),
        marker=dict(size=10, color="#3D6BE8", symbol="diamond",
                    line=dict(color="#fff",width=2)),
        fill="tozeroy", fillcolor="rgba(61,107,232,0.07)",
        hovertemplate="Canggu · %{x}: %{y:.1f}%<extra></extra>",
    ))
    fig_area.add_trace(go.Scatter(
        x=MONTHS_ORDER, y=seminyak_monthly, name="Seminyak",
        mode="lines+markers",
        line=dict(color="#f97316", width=3),
        marker=dict(size=10, color="#f97316", symbol="circle",
                    line=dict(color="#fff",width=2)),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.07)",
        hovertemplate="Seminyak · %{x}: %{y:.1f}%<extra></extra>",
    ))
    # annotation avg
    fig_area.add_hline(y=canggu_avg, line_dash="dot", line_color="#3D6BE8",
                       line_width=1, annotation_text=f"Canggu avg {canggu_avg}%",
                       annotation_font_size=10, annotation_font_color="#3D6BE8")
    fig_area.add_hline(y=seminyak_avg, line_dash="dot", line_color="#f97316",
                       line_width=1, annotation_text=f"Seminyak avg {seminyak_avg}%",
                       annotation_font_size=10, annotation_font_color="#f97316")

    fig_area.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#f0f6ff",
        font_family="Sora", height=300,
        margin=dict(l=0,r=10,t=10,b=30),
        legend=dict(orientation="h", y=1.1, x=0, font_size=11, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickfont_size=11, linecolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%",
                   range=[0,105], tickfont_size=11),
    )
    st.plotly_chart(fig_area, width='stretch')

with col_chart_r:
    st.markdown("<div class='section-label'><span>03</span> · Distribusi Performa — Radar per Vila</div>", unsafe_allow_html=True)

    categories = MONTHS_ORDER + [MONTHS_ORDER[0]]
    fig_radar = go.Figure()
    for vn in villa_names:
        vc = get_color(vn)
        r,g,b = safe_hex(vc)
        vals_r = [FORECAST_2026[vn][m] for m in MONTHS_ORDER]
        vals_r.append(vals_r[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_r, theta=categories,
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.07)",
            line=dict(color=f"rgba({r},{g},{b},0.9)", width=2),
            name=vn.replace(" Villas",""),
            hovertemplate="%{theta}: %{r:.1f}%<extra>" + vn.replace(" Villas","") + "</extra>",
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, range=[0,100], ticksuffix="%",
                            tickfont_size=9, gridcolor="#e2e8f0", linecolor="#e2e8f0"),
            angularaxis=dict(tickfont_size=11, linecolor="#e2e8f0", gridcolor="#f1f5f9"),
        ),
        plot_bgcolor="#ffffff", paper_bgcolor="#f0f6ff",
        font_family="Sora", height=300,
        margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    font_size=10, bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    st.plotly_chart(fig_radar, width='stretch')

st.markdown("<hr class='rule'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP TABLE — Proyeksi per Vila per Bulan
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label anim-4'><span>04</span> · Heatmap Proyeksi Hunian — Jan–Jun 2026 per Vila</div>", unsafe_allow_html=True)

def build_heat_cell(v):
    bg, fg = heat_bg(v)
    label  = "—" if v == 0 else f"{v:.0f}%"
    return f"<div class='heat-cell' style='background:{bg};color:{fg}'>{label}</div>"

heatmap_rows = ""
for vn in villa_names:
    vc    = get_color(vn)
    area  = VILLA_INSIGHTS[vn]["area"]
    fc    = FORECAST_2026[vn]
    avg   = fc["avg"]
    cells = "".join([f"<td style='text-align:center'>{build_heat_cell(fc[m])}</td>" for m in MONTHS_ORDER])
    avg_bg, avg_fg = heat_bg(avg)

    # historical mean from computed metrics
    hist_m = hist_metrics[vn]["mean"]
    delta  = avg - hist_m
    ds     = f"+{delta:.1f}pp" if delta >= 0 else f"{delta:.1f}pp"
    dc     = "#16a34a" if delta >= 0 else "#dc2626"

    heatmap_rows += f"""
    <tr>
      <td>
        <div style='display:flex;align-items:center;gap:8px'>
          <span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{vc};flex-shrink:0'></span>
          <div>
            <div style='font-weight:700;font-size:13px;color:#0f172a'>{vn}</div>
            <div style='font-size:10px;color:#94a3b8;font-family:DM Mono,monospace;text-transform:uppercase'>{area}</div>
          </div>
        </div>
      </td>
      {cells}
      <td style='text-align:center'>
        <div class='heat-cell' style='background:{avg_bg};color:{avg_fg};width:62px'>{avg:.1f}%</div>
      </td>
      <td style='text-align:center'>
        <span style='font-family:DM Mono,monospace;font-size:12px;font-weight:600;color:{dc}'>{ds}</span>
        <div style='font-size:9px;color:#94a3b8;font-family:DM Mono,monospace'>vs historis</div>
      </td>
    </tr>"""

month_headers = "".join([f"<th style='text-align:center'>{m}</th>" for m in MONTHS_ORDER])
st.markdown(f"""
<table class='heatmap-table'>
  <thead><tr>
    <th style='width:200px'>Vila</th>
    {month_headers}
    <th style='text-align:center'>Avg</th>
    <th style='text-align:center'>Δ Hist</th>
  </tr></thead>
  <tbody>{heatmap_rows}</tbody>
</table>
""", unsafe_allow_html=True)

# Legend
st.markdown("""
<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;margin-bottom:24px'>
  <span style='font-size:10px;color:#94a3b8;font-family:DM Mono,monospace;align-self:center'>Skala:</span>
  <span style='background:#166534;color:#dcfce7;font-size:10px;padding:3px 10px;border-radius:4px;font-family:DM Mono,monospace'>≥85%</span>
  <span style='background:#15803d;color:#f0fdf4;font-size:10px;padding:3px 10px;border-radius:4px;font-family:DM Mono,monospace'>70–84%</span>
  <span style='background:#ca8a04;color:#fefce8;font-size:10px;padding:3px 10px;border-radius:4px;font-family:DM Mono,monospace'>55–69%</span>
  <span style='background:#ea580c;color:#fff7ed;font-size:10px;padding:3px 10px;border-radius:4px;font-family:DM Mono,monospace'>40–54%</span>
  <span style='background:#dc2626;color:#fef2f2;font-size:10px;padding:3px 10px;border-radius:4px;font-family:DM Mono,monospace'>20–39%</span>
  <span style='background:#991b1b;color:#fce7f3;font-size:10px;padding:3px 10px;border-radius:4px;font-family:DM Mono,monospace'>&lt;20%</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VILLA PERFORMANCE CARDS per Area
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'><span>05</span> · Performa Vila per Kawasan — Historis &amp; Proyeksi</div>", unsafe_allow_html=True)

for area in AREA_ORDER:
    area_villas = [vn for vn in villa_names if VILLA_INSIGHTS[vn]["area"] == area]
    pill_cls    = "canggu" if area == "Canggu" else "seminyak"
    area_avg_fc = np.mean([FORECAST_2026[v]["avg"] for v in area_villas])

    st.markdown(f"""
    <div class='area-header'>
      <span class='area-pill {pill_cls}'>📍 {area}</span>
      <span style='font-size:12px;color:#64748b;font-family:DM Mono,monospace'>
        {len(area_villas)} vila · Avg proyeksi {area_avg_fc:.1f}%
      </span>
      <div class='area-divider'></div>
    </div>
    """, unsafe_allow_html=True)

    for vn in area_villas:
        vc      = get_color(vn)
        fc      = FORECAST_2026[vn]
        hm      = hist_metrics[vn]
        avg_fc  = fc["avg"]
        avg_oc  = occ_color(avg_fc)
        meta    = SARIMA_META_STATIC[vn]
        trend   = hm["trend"]
        ts      = f"+{trend:.1f}pp YoY" if trend >= 0 else f"{trend:.1f}pp YoY"
        tc      = "#16a34a" if trend >= 0 else "#dc2626"
        bar_w   = int(avg_fc)

        # spark-line mini
        spark_vals = [fc[m] for m in MONTHS_ORDER]
        spark_min, spark_max = min(spark_vals), max(spark_vals)

        st.markdown(f"""
        <div class='villa-row' style='--vc:{vc}'>
          <div>
            <div class='villa-name'>{vn}</div>
            <div class='villa-area'>{area} · {VILLA_INSIGHTS[vn]["elastisitas"]}</div>
            <div style='margin-top:6px'>
              <span style='font-family:DM Mono,monospace;font-size:9px;
                background:#f1f5f9;border:1px solid #e2e8f0;border-radius:4px;
                padding:2px 6px;color:#64748b'>{meta["order"]}</span>
            </div>
          </div>
          <div>
            <div style='font-family:DM Mono,monospace;font-size:26px;
                 font-weight:700;color:{avg_oc};line-height:1'>{avg_fc:.1f}%</div>
            <div style='font-size:10px;color:#94a3b8;margin-bottom:6px;font-family:DM Mono,monospace'>
              avg proyeksi Sem I 2026
            </div>
            <div class='mini-bar-wrap' style='max-width:220px'>
              <div class='mini-bar' style='width:{bar_w}%;background:{avg_oc}'></div>
            </div>
          </div>
          <div>
            <div class='villa-stat-label'>Hist. Mean</div>
            <div class='villa-stat-val'>{hm["mean"]:.1f}%</div>
          </div>
          <div>
            <div class='villa-stat-label'>Peak Hist.</div>
            <div class='villa-stat-val'>{hm["peak_mean"]:.1f}%</div>
          </div>
          <div>
            <div class='villa-stat-label'>YoY Trend</div>
            <div style='font-family:DM Mono,monospace;font-size:14px;font-weight:600;color:{tc}'>{ts}</div>
          </div>
          <div>
            <div class='villa-stat-label'>RMSE · MAPE</div>
            <div class='villa-stat-val'>{meta["rmse"]:.2f} · {meta["mape"]:.1f}%</div>
            <div style='font-size:9px;color:#94a3b8;font-family:DM Mono,monospace;margin-top:2px'>
              test set historis
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Quick Nav ─────────────────────────────────────────────────────────────────
st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>Navigasi Cepat</div>", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("🔮  Prediksi SARIMA Detail", width='stretch'):
        st.switch_page("pages/2_Prediksi.py")
with nav2:
    if st.button("📤  Upload Data Terbaru", width='stretch'):
        st.switch_page("pages/3_Upload.py")
with nav3:
    if st.button("👤  Manajemen User", width='stretch'):
        st.switch_page("pages/4_Users.py")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 12px;
    color:#cbd5e1;font-size:10px;font-family:DM Mono,monospace;letter-spacing:.1em'>
  VILLAS R US ANALYTICS · DASHBOARD UTAMA · SARIMA FORECAST 2026 · PT BALI CIPTA VILA MANDIRI
</div>
""", unsafe_allow_html=True)