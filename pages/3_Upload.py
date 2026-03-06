import streamlit as st
st.cache = st.cache_data
import pandas as pd
from datetime import datetime

from database import (
    get_villas, insert_occupancy_bulk, insert_financial_bulk,
    log_upload, get_upload_logs, get_data_summary, get_conn
)
from utils.data_processor import process_occupancy_csv, process_financial_csv
from utils.auth import get_cookie_manager, set_session, load_from_cookie
from utils.sidebar import render_sidebar

# ─── COOKIES ─────────────────────────────────────────────────────────────────
cookies = get_cookie_manager()

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

# ─── VILLA CRUD HELPERS ─────────────────────────────────────────────────────────
def add_villa(villa_code: str, villa_name: str, area: str, description: str=""):
    conn = get_conn()
    if not conn:
        return False, "Koneksi database gagal."
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM villas WHERE villa_code=%s", (villa_code,))
        if cur.fetchone():
            return False, f"Kode villa '{villa_code}' sudah digunakan."
        cur.execute(
            "INSERT INTO villas (villa_code, villa_name, area, description) VALUES (%s,%s,%s,%s)",
            (villa_code.upper(), villa_name, area, description)
        )
        conn.commit(); cur.close(); conn.close()
        return True, f"Villa '{villa_name}' berhasil ditambahkan."
    except Exception as e:
        return False, str(e)


def update_villa(villa_id: int, villa_name: str, area: str, description: str):
    conn = get_conn()
    if not conn:
        return False, "Koneksi database gagal."
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE villas SET villa_name=%s, area=%s, description=%s WHERE id=%s",
            (villa_name, area, description, villa_id)
        )
        conn.commit(); cur.close(); conn.close()
        return True, "Villa berhasil diperbarui."
    except Exception as e:
        return False, str(e)


def delete_villa(villa_id: int, villa_code: str):
    conn = get_conn()
    if not conn:
        return False, "Koneksi database gagal."
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM occupancy_data WHERE villa_code=%s", (villa_code,))
        occ_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM financial_data WHERE villa_code=%s", (villa_code,))
        fin_count = cur.fetchone()[0]
        if occ_count + fin_count > 0:
            return False, f"Villa masih memiliki {occ_count} data ocupansi dan {fin_count} data finansial. Hapus data terlebih dahulu."
        cur.execute("DELETE FROM villas WHERE id=%s", (villa_id,))
        conn.commit(); cur.close(); conn.close()
        return True, "Villa berhasil dihapus."
    except Exception as e:
        return False, str(e)


# ─── CSS (LIGHT THEME) ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background: #f0f6ff; color: #0f172a; }
  #MainMenu, footer { visibility: hidden; }

  [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
  [data-testid="stSidebarNav"] { display: none !important; }

  .section-title {
    font-size:11px; font-weight:700; color:#94a3b8;
    letter-spacing:.12em; text-transform:uppercase;
    margin-bottom:10px; padding-bottom:6px;
    border-bottom:1px solid #e2e8f0;
  }
  .stat-box {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 14px 18px; text-align: center;
    box-shadow: 0 2px 8px rgba(3,105,161,0.06);
  }
  .stat-num   { font-size:22px; font-weight:800; font-family:'DM Mono',monospace; }
  .stat-label { font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; }

  .villa-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px 18px; margin-bottom: 10px;
    box-shadow: 0 2px 6px rgba(3,105,161,0.05);
    transition: border-color .2s, box-shadow .2s;
  }
  .villa-card:hover { border-color: #38bdf8; box-shadow: 0 4px 14px rgba(3,105,161,0.10); }

  .badge-success { background:#dcfce7; color:#166534; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-warn    { background:#fef9c3; color:#854d0e; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-fail    { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }

  .area-tag {
    display:inline-block; background:#eff6ff; color:#1d4ed8;
    padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600;
  }
  .code-tag {
    display:inline-block; background:#f1f5f9; color:#475569;
    padding:2px 8px; border-radius:6px; font-size:11px;
    font-family:'DM Mono',monospace; font-weight:600;
  }

  /* Info panels */
  .info-panel-blue {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 14px 16px; font-size: 12px; color: #334155;
  }
  .info-panel-purple {
    background: #f5f3ff; border: 1px solid #ddd6fe;
    border-radius: 10px; padding: 14px 16px; font-size: 12px; color: #334155;
  }
  .info-panel-amber {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 14px 16px; font-size: 12px; color: #334155;
  }
  .danger-panel {
    background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 10px; padding: 12px 14px; font-size: 12px; color: #991b1b; margin-bottom: 10px;
  }

  /* Inputs */
  input[type="text"], input[type="password"], textarea {
    background: #f8fafc !important; border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important; color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important; caret-color: #0369a1 !important;
  }
  input[type="text"]:focus, input[type="password"]:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
  }
  input::placeholder { color:#94a3b8 !important; -webkit-text-fill-color:#94a3b8 !important; }
  label, .stTextInput label { color:#334155 !important; font-weight:600 !important; font-size:13px !important; }

  .stButton > button { border-radius: 8px !important; font-weight: 600 !important; font-size: 13px !important; }

  div[data-testid="stForm"] {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 14px; padding: 20px;
    box-shadow: 0 2px 8px rgba(3,105,161,0.05);
  }

  /* Sidebar buttons */
  section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important; color: #334155 !important;
    border: 1px solid #e2e8f0 !important; box-shadow: none !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f0f6ff !important; border-color: #38bdf8 !important; color: #0369a1 !important;
  }
</style>
""", unsafe_allow_html=True)


# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:16px'>
  <span style='font-size:22px;font-weight:800;color:#0f172a'>📁 Upload & Manajemen Data</span>
  <span style='font-size:13px;color:#94a3b8;margin-left:12px'>Import CSV dari Beds24 · Auto Cleaning</span>
</div>
<hr style='border:none;border-top:1px solid #e2e8f0;margin:0 0 16px'>
""", unsafe_allow_html=True)

# ─── LOAD VILLAS ────────────────────────────────────────────────────────────────
df_villas = get_villas()
if df_villas is None or df_villas.empty:
    villa_options = {}
    st.warning("⚠️ Belum ada villa terdaftar. Tambahkan villa terlebih dahulu di tab **🏠 Kelola Villa**.")
else:
    villa_options = {row["villa_name"]: row["villa_code"] for _, row in df_villas.iterrows()}

# ─── DATA SUMMARY ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Status Data Saat Ini</div>", unsafe_allow_html=True)
summary = get_data_summary()
if summary is not None and not summary.empty:
    cols = st.columns(min(len(summary), 6))
    for idx, (_, row) in enumerate(summary.iterrows()):
        occ_rows = int(row.get("occ_rows", 0) or 0)
        fin_rows = int(row.get("fin_rows", 0) or 0)
        occ_from = str(row.get("occ_from", "-") or "-")
        cols[idx % len(cols)].markdown(f"""
        <div class='stat-box'>
          <div style='font-size:12px;font-weight:700;color:#0f172a;margin-bottom:4px'>{row["villa_name"]}</div>
          <span class='area-tag'>{row["area"]}</span>
          <hr style='border:none;border-top:1px solid #f1f5f9;margin:10px 0'>
          <div class='stat-num' style='color:#0369a1;font-size:18px'>{occ_rows:,}</div>
          <div class='stat-label'>Occ Rows</div>
          <div class='stat-num' style='color:#6d28d9;font-size:18px;margin-top:6px'>{fin_rows:,}</div>
          <div class='stat-label'>Fin Rows</div>
          <div style='font-size:10px;color:#94a3b8;margin-top:6px'>{occ_from[:10] if occ_from!="-" else "–"}</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("Belum ada data. Upload file CSV pertama Anda.")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
utab1, utab2, utab3, utab4 = st.tabs([
    "📊 Upload Ocupansi",
    "💰 Upload Finansial",
    "🏠 Kelola Villa",
    "🗂️ Log Upload",
])

# ════════════════ TAB 1: UPLOAD OCCUPANCY ════════════════
with utab1:
    st.markdown("<div class='section-title'>Import Data Ocupansi (Beds24 CSV)</div>", unsafe_allow_html=True)

    if not villa_options:
        st.error("Tidak ada villa terdaftar. Tambahkan villa di tab **🏠 Kelola Villa** terlebih dahulu.")
    else:
        oc1, oc2 = st.columns([2, 3])
        with oc1:
            sel_villa_occ = st.selectbox("🏠 Pilih Villa", list(villa_options.keys()), key="up_occ_villa")
            villa_code_occ = villa_options[sel_villa_occ]
            st.markdown(f"<div style='font-size:11px;color:#64748b;margin-bottom:8px'>Kode: <span class='code-tag'>{villa_code_occ}</span></div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='info-panel-blue'>
              <b style='color:#0369a1'>📋 Format CSV yang Diterima:</b><br><br>
              Kolom wajib: <code>Date</code>, <code>Occupancy Total</code><br><br>
              Kolom opsional: <code>Arrivals</code>, <code>Arriving Guests</code>,
              <code>Departures</code>, <code>Departing Guests</code>, <code>Stay Through</code>,
              <code>Staying Guests</code>, <code>Booked</code>, <code>Booked Guests</code>,
              <code>Available</code>, <code>Black</code><br><br>
              Nilai occupancy &gt;100% akan di-clip ke 100% secara otomatis.
            </div>
            """, unsafe_allow_html=True)

        with oc2:
            uploaded_occ = st.file_uploader(
                "Drag & drop atau klik untuk upload CSV Ocupansi",
                type=["csv"], key="file_occ",
                help="File CSV ekspor dari Beds24 > Reports > Occupancy"
            )

            if uploaded_occ is not None:
                with st.spinner("🔄 Memproses dan membersihkan data..."):
                    result = process_occupancy_csv(uploaded_occ, villa_code_occ)

                stats = result["stats"]
                errors = result["errors"]
                df_prev = result["df_preview"]

                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#0369a1'>{stats.get('total',0):,}</div><div class='stat-label'>Total Baris</div></div>", unsafe_allow_html=True)
                s2.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#16a34a'>{stats.get('imported',0):,}</div><div class='stat-label'>Siap Import</div></div>", unsafe_allow_html=True)
                s3.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#d97706'>{stats.get('skipped',0):,}</div><div class='stat-label'>Dilewati</div></div>", unsafe_allow_html=True)
                s4.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#db2777'>{stats.get('anomalies_clipped',0):,}</div><div class='stat-label'>Anomali Di-clip</div></div>", unsafe_allow_html=True)

                if errors:
                    with st.expander(f"⚠️ {len(errors)} peringatan"):
                        for e in errors[:10]: st.warning(e)

                st.markdown("**Preview data setelah cleaning:**")
                st.dataframe(df_prev, use_container_width='stretch', height=200)

                if result["records"]:
                    if st.button("✅ Konfirmasi & Import ke Database", key="btn_import_occ",
                                 type="primary", use_container_width='stretch'):
                        with st.spinner("Mengimport data ke MySQL..."):
                            ok = insert_occupancy_bulk(result["records"])
                            status = "success" if ok else "failed"
                            log_upload(
                                filename=uploaded_occ.name, file_type="occupancy",
                                villa_code=villa_code_occ,
                                rows_total=stats.get("total", 0),
                                rows_imported=stats.get("imported", 0) if ok else 0,
                                rows_skipped=stats.get("skipped", 0),
                                status=status, user_id=st.session_state.get("user_id"),
                                notes=f"Anomali di-clip: {stats.get('anomalies_clipped',0)}"
                            )
                        if ok:
                            st.success(f"✅ Berhasil mengimport {stats.get('imported',0):,} baris data ocupansi untuk **{sel_villa_occ}**!")
                            st.cache_data.clear(); st.rerun()
                        else:
                            st.error("❌ Gagal mengimport data. Cek koneksi database.")
                else:
                    st.error("Tidak ada data valid untuk diimport.")

# ════════════════ TAB 2: UPLOAD FINANCIAL ════════════════
with utab2:
    st.markdown("<div class='section-title'>Import Data Finansial (Beds24 CSV)</div>", unsafe_allow_html=True)

    if not villa_options:
        st.error("Tidak ada villa terdaftar. Tambahkan villa di tab **🏠 Kelola Villa** terlebih dahulu.")
    else:
        fc1, fc2 = st.columns([2, 3])
        with fc1:
            sel_villa_fin = st.selectbox("🏠 Pilih Villa", list(villa_options.keys()), key="up_fin_villa")
            villa_code_fin = villa_options[sel_villa_fin]
            st.markdown(f"<div style='font-size:11px;color:#64748b;margin-bottom:8px'>Kode: <span class='code-tag'>{villa_code_fin}</span></div>", unsafe_allow_html=True)

            st.markdown("""
            <div class='info-panel-purple'>
              <b style='color:#6d28d9'>📋 Format CSV yang Diterima:</b><br><br>
              Kolom utama: <code>Date</code>, <code>Average Daily Revenue IDR</code>
              (atau <code>ADR</code>), <code>Daily Revenue IDR</code>,
              <code>Room Revenue IDR</code>, <code>RevPAR</code>,
              <code>Revenue per Guest IDR</code><br><br>
              <b>Kebijakan outlier:</b> Seluruh data <b>tetap disimpan</b>.
              ADR melampaui batas IQR×3.5 hanya di-flag sebagai
              <code>is_outlier_adr=1</code>.<br><br>
              <b>Zero revenue</b> dibedakan menjadi dua: vila kosong
              (<code>is_empty_villa</code>) vs vila terisi tanpa ADR
              (<code>is_adr_missing</code>).
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class='info-panel-amber' style='margin-top:10px'>
              <b style='color:#b45309'>📐 Kolom Flag yang Dihasilkan:</b><br><br>
              <code>is_empty_villa</code> — guests = 0 (kosong murni)<br>
              <code>is_adr_missing</code> — guests &gt; 0 &amp; ADR = 0<br>
              <code>is_outlier_adr</code> — ADR melampaui IQR×3.5<br>
              <code>for_modeling</code> — subset bersih SARIMA
            </div>
            """, unsafe_allow_html=True)

        with fc2:
            uploaded_fin = st.file_uploader(
                "Drag & drop atau klik untuk upload CSV Finansial",
                type=["csv"], key="file_fin",
                help="File CSV ekspor dari Beds24 > Reports > Revenue"
            )

            if uploaded_fin is not None:
                with st.spinner("🔄 Memproses dan membersihkan data..."):
                    result_f = process_financial_csv(uploaded_fin, villa_code_fin)

                stats_f = result_f["stats"]
                errors_f = result_f["errors"]
                df_prev_f = result_f["df_preview"]

                # ── Baris 1: ringkasan volume ──
                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#6d28d9'>{stats_f.get('total',0):,}</div><div class='stat-label'>Total Baris</div></div>", unsafe_allow_html=True)
                s2.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#16a34a'>{stats_f.get('imported',0):,}</div><div class='stat-label'>Diimport</div></div>", unsafe_allow_html=True)
                s3.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#0369a1'>{stats_f.get('for_modeling',0):,}</div><div class='stat-label'>For Modeling</div></div>", unsafe_allow_html=True)
                s4.markdown(f"<div class='stat-box'><div class='stat-num' style='color:#d97706'>{stats_f.get('duplicates_removed',0):,}</div><div class='stat-label'>Duplikat Hapus</div></div>", unsafe_allow_html=True)

                # ── Baris 2: flag detail ──
                st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
                f1, f2, f3 = st.columns(3)

                empty_v   = stats_f.get('empty_villa_days', 0)
                adr_miss  = stats_f.get('adr_missing_days', 0)
                outliers  = stats_f.get('outliers_flagged', 0)
                iqr_upper = stats_f.get('iqr_upper')
                iqr_label = f"upper ≈ {iqr_upper:,.0f}" if iqr_upper else "–"

                f1.markdown(f"""
                <div class='stat-box'>
                  <div class='stat-num' style='color:#94a3b8'>{empty_v:,}</div>
                  <div class='stat-label'>🏚️ Vila Kosong</div>
                  <div style='font-size:10px;color:#94a3b8;margin-top:4px'>is_empty_villa</div>
                </div>""", unsafe_allow_html=True)

                f2.markdown(f"""
                <div class='stat-box'>
                  <div class='stat-num' style='color:#f59e0b'>{adr_miss:,}</div>
                  <div class='stat-label'>📭 ADR Kosong</div>
                  <div style='font-size:10px;color:#94a3b8;margin-top:4px'>is_adr_missing</div>
                </div>""", unsafe_allow_html=True)

                f3.markdown(f"""
                <div class='stat-box'>
                  <div class='stat-num' style='color:#dc2626'>{outliers:,}</div>
                  <div class='stat-label'>🚩 ADR Outlier (flag)</div>
                  <div style='font-size:10px;color:#94a3b8;margin-top:4px'>{iqr_label} IDR</div>
                </div>""", unsafe_allow_html=True)

                if errors_f:
                    with st.expander(f"ℹ️ {len(errors_f)} catatan proses"):
                        for e in errors_f[:10]: st.info(e)

                st.markdown("**Preview data setelah cleaning (termasuk kolom flag):**")
                st.dataframe(df_prev_f, use_container_width='stretch', height=220)

                if result_f["records"]:
                    if st.button("✅ Konfirmasi & Import ke Database", key="btn_import_fin",
                                 type="primary", use_container_width='stretch'):
                        with st.spinner("Mengimport data ke MySQL..."):
                            ok_f = insert_financial_bulk(result_f["records"])
                            st_f = "success" if ok_f else "failed"
                            log_upload(
                                filename=uploaded_fin.name, file_type="financial",
                                villa_code=villa_code_fin,
                                rows_total=stats_f.get("total", 0),
                                rows_imported=stats_f.get("imported", 0) if ok_f else 0,
                                rows_skipped=stats_f.get("skipped", 0),
                                status=st_f, user_id=st.session_state.get("user_id"),
                                notes=(
                                    f"Flag: outlier={stats_f.get('outliers_flagged',0)}, "
                                    f"empty={stats_f.get('empty_villa_days',0)}, "
                                    f"adr_missing={stats_f.get('adr_missing_days',0)}, "
                                    f"for_modeling={stats_f.get('for_modeling',0)}"
                                )
                            )
                        if ok_f:
                            st.success(
                                f"✅ Berhasil mengimport **{stats_f.get('imported',0):,}** baris untuk **{sel_villa_fin}** — "
                                f"**{stats_f.get('for_modeling',0):,}** baris siap pemodelan, "
                                f"**{stats_f.get('outliers_flagged',0)}** outlier ADR di-flag (data tetap ada)."
                            )
                            st.cache_data.clear(); st.rerun()
                        else:
                            st.error("❌ Gagal mengimport. Cek koneksi database.")
                else:
                    st.error("Tidak ada data valid untuk diimport.")

# ════════════════ TAB 3: KELOLA VILLA ════════════════
with utab3:
    is_admin = st.session_state.get("role") == "admin"

    st.markdown("<div class='section-title'>Daftar Villa Terdaftar</div>", unsafe_allow_html=True)

    col_ref, _ = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh", key="refresh_villa", use_container_width='stretch'):
            st.cache_data.clear(); st.rerun()

    df_v = get_villas()
    if df_v is None or df_v.empty:
        st.info("Belum ada villa terdaftar.")
    else:
        for _, row in df_v.iterrows():
            vid = row.get("id")
            vcode = row.get("villa_code", "")
            vname = row.get("villa_name", "")
            varea = row.get("area", "")
            vdesc = row.get("description", "") or ""

            st.markdown(f"""
            <div class='villa-card'>
              <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>
                <div style='font-size:26px'>🏝️</div>
                <div style='flex:1'>
                  <div style='font-size:14px;font-weight:700;color:#0f172a'>{vname}</div>
                  <div style='font-size:12px;color:#64748b;margin-top:2px'>{vdesc if vdesc else "–"}</div>
                </div>
                <div><span class='code-tag'>{vcode}</span></div>
                <div><span class='area-tag'>{varea}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if is_admin:
                with st.expander(f"⚙️ Kelola Villa: {vname}"):
                    ec1, ec2 = st.columns([1, 1])

                    with ec1:
                        st.markdown("**✏️ Edit Villa**")
                        with st.form(key=f"edit_villa_{vid}"):
                            e_name = st.text_input("Nama Villa", value=vname, key=f"vn_{vid}")
                            e_area = st.text_input("Area / Lokasi", value=varea, key=f"va_{vid}")
                            e_desc = st.text_input("Deskripsi (opsional)", value=vdesc, key=f"vd_{vid}")
                            if st.form_submit_button("💾 Simpan", use_container_width='stretch'):
                                ok, msg = update_villa(vid, e_name, e_area, e_desc)
                                if ok: st.success(msg); st.cache_data.clear(); st.rerun()
                                else: st.error(msg)

                    with ec2:
                        st.markdown("**🗑️ Hapus Villa**")
                        st.markdown("<div class='danger-panel'>⚠️ Villa hanya bisa dihapus jika tidak memiliki data ocupansi maupun finansial.</div>", unsafe_allow_html=True)
                        with st.form(key=f"del_villa_{vid}"):
                            confirm_v = st.text_input(f"Ketik '{vcode}' untuk konfirmasi", key=f"dvc_{vid}")
                            if st.form_submit_button("🗑️ Hapus Villa", use_container_width='stretch', type="primary"):
                                if confirm_v.strip().upper() == vcode.upper():
                                    ok, msg = delete_villa(vid, vcode)
                                    if ok: st.success(msg); st.cache_data.clear(); st.rerun()
                                    else: st.error(msg)
                                else:
                                    st.error("Konfirmasi kode villa tidak cocok.")

    if is_admin:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>➕ Tambah Villa Baru</div>", unsafe_allow_html=True)

        al, ar = st.columns([1, 1])
        with al:
            with st.form("form_add_villa", clear_on_submit=True):
                new_vcode = st.text_input("Kode Villa *", placeholder="cth: VLR-04 (unik, huruf kapital)")
                new_vname = st.text_input("Nama Villa *", placeholder="cth: Villa Lumbung Rose")
                new_varea = st.text_input("Area / Lokasi *", placeholder="cth: Ubud, Seminyak, Canggu")
                new_vdesc = st.text_input("Deskripsi (opsional)", placeholder="cth: Villa 3 kamar dengan private pool")

                if st.form_submit_button("➕ Tambah Villa", use_container_width='stretch', type="primary"):
                    if not all([new_vcode, new_vname, new_varea]):
                        st.error("Kode, nama, dan area villa wajib diisi.")
                    elif " " in new_vcode.strip():
                        st.error("Kode villa tidak boleh mengandung spasi. Gunakan tanda hubung, cth: VLR-04")
                    else:
                        ok, msg = add_villa(new_vcode.strip(), new_vname.strip(), new_varea.strip(), new_vdesc.strip())
                        if ok:
                            st.success(f"✅ {msg}")
                            st.cache_data.clear(); st.rerun()
                        else:
                            st.error(f"❌ {msg}")

        with ar:
            st.markdown("""
            <div class='info-panel-blue'>
              <div style='font-weight:700;color:#0369a1;margin-bottom:12px'>💡 Panduan Kode Villa</div>
              <ul style='margin:0;padding-left:16px;line-height:2;color:#475569'>
                <li>Kode harus <b>unik</b> dan tidak bisa diubah setelah villa punya data</li>
                <li>Gunakan format singkatan + nomor, cth: <code>VLR-01</code>, <code>SMYK-02</code></li>
                <li>Tidak boleh mengandung spasi — gunakan tanda hubung (<code>-</code>)</li>
                <li>Kode akan otomatis diubah ke <b>HURUF KAPITAL</b></li>
                <li>Area diisi nama lokasi, cth: Ubud, Seminyak, Canggu, Nusa Dua</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Hanya admin yang dapat menambah, mengedit, atau menghapus villa.")

# ════════════════ TAB 4: LOG UPLOAD ════════════════
with utab4:
    st.markdown("<div class='section-title'>Riwayat Upload</div>", unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width='stretch', key="refresh_log"):
            st.rerun()

    logs = get_upload_logs()
    if logs is not None and not logs.empty:
        display_cols = ["uploaded_at", "full_name", "filename", "file_type", "villa_code",
                        "rows_total", "rows_imported", "rows_skipped", "status", "notes"]
        disp = logs[[c for c in display_cols if c in logs.columns]].copy()
        disp["uploaded_at"] = pd.to_datetime(disp["uploaded_at"]).dt.strftime("%d %b %Y %H:%M")

        def badge(s):
            if s == "success": return "✅ Success"
            if s == "partial": return "⚠️ Partial"
            return "❌ Failed"

        disp["status"] = disp["status"].apply(badge)

        st.dataframe(disp, use_container_width='stretch', height=400,
                     column_config={
                         "uploaded_at": st.column_config.TextColumn("Waktu Upload"),
                         "full_name": st.column_config.TextColumn("Diupload Oleh"),
                         "filename": st.column_config.TextColumn("Nama File"),
                         "file_type": st.column_config.TextColumn("Tipe"),
                         "villa_code": st.column_config.TextColumn("Villa"),
                         "rows_total": st.column_config.NumberColumn("Total"),
                         "rows_imported": st.column_config.NumberColumn("Imported"),
                         "rows_skipped": st.column_config.NumberColumn("Skipped"),
                         "status": st.column_config.TextColumn("Status"),
                         "notes": st.column_config.TextColumn("Catatan"),
                     })
    else:
        st.info("Belum ada riwayat upload.")

    # ── Danger Zone: Hapus Data per Villa (admin only) ──
    if st.session_state.get("role") == "admin" and villa_options:
        st.markdown("<hr style='border:none;border-top:1px solid #fecaca;margin:20px 0'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='color:#dc2626'>⚠️ Danger Zone – Hapus Data</div>", unsafe_allow_html=True)

        d1, d2, d3 = st.columns([2, 2, 2])
        with d1:
            del_villa = st.selectbox("Villa", list(villa_options.keys()), key="del_villa")
        with d2:
            del_type = st.selectbox("Tipe Data", ["occupancy", "financial", "keduanya"], key="del_type")
        with d3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Hapus Data", type="secondary", use_container_width='stretch', key="btn_del"):
                vc_del = villa_options[del_villa]
                conn = get_conn()
                if conn:
                    cur = conn.cursor()
                    if del_type in ["occupancy", "keduanya"]:
                        cur.execute("DELETE FROM occupancy_data WHERE villa_code=%s", (vc_del,))
                    if del_type in ["financial", "keduanya"]:
                        cur.execute("DELETE FROM financial_data WHERE villa_code=%s", (vc_del,))
                    conn.commit(); cur.close(); conn.close()
                    st.success(f"✅ Data {del_type} untuk **{del_villa}** berhasil dihapus.")
                    st.cache_data.clear(); st.rerun()