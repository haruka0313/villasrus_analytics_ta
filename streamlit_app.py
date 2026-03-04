import streamlit as st
import hashlib
from dotenv import load_dotenv
from database import init_db, get_user_by_credentials, run_query
from utils.auth import get_cookie_manager, set_session, save_to_cookie, load_from_cookie

load_dotenv()

# ─── COOKIE — WAJIB PALING ATAS ──────────────────────────────────────────────
cookies = get_cookie_manager()
if not cookies.ready():
    st.stop()

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Villas R Us · Login",
    page_icon="🏝️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background: #f0f6ff; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stSidebar"]        { display:none !important; }
  [data-testid="stSidebarNav"]     { display:none !important; }
  [data-testid="collapsedControl"] { display:none !important; }
  .auth-wrap { max-width:460px; margin:0 auto; padding:32px 0; }
  .auth-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:20px;
    padding:36px 36px 28px; box-shadow:0 8px 32px rgba(3,105,161,0.10);
  }
  .brand-logo  { font-size:42px; text-align:center; margin-bottom:8px; }
  .brand-title { font-size:22px; font-weight:800; color:#0f172a; text-align:center; }
  .brand-sub   { font-size:12px; color:#94a3b8; text-align:center; margin-bottom:24px; letter-spacing:.06em; }
  input[type="text"], input[type="password"] {
    background:#f8fafc !important; border:1px solid #cbd5e1 !important;
    border-radius:10px !important; color:#0f172a !important;
    -webkit-text-fill-color:#0f172a !important;
  }
  .stButton > button {
    background:linear-gradient(135deg,#0369a1,#38bdf8) !important;
    color:#fff !important; font-weight:700 !important; border:none !important;
    border-radius:10px !important; height:44px !important; width:100% !important;
  }
  .hint-box {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
    padding:12px 16px; font-size:12px; color:#3b82f6; margin-top:14px;
  }
  .hint-box code { background:#dbeafe; padding:2px 6px; border-radius:4px; color:#1d4ed8; }
  .reg-info {
    background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
    padding:12px 16px; font-size:12px; color:#15803d; margin-top:14px;
  }
  .stTabs [data-baseweb="tab-list"] { gap:8px; background:#f1f5f9; border-radius:12px; padding:4px; }
  .stTabs [aria-selected="true"] { background:#ffffff !important; color:#0369a1 !important; }
</style>
""", unsafe_allow_html=True)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def register_user(username, full_name, password):
    existing = run_query("SELECT id FROM users WHERE username=%s", (username,))
    if existing is not None and not existing.empty:
        return False, f"Username '@{username}' sudah digunakan."
    ok = run_query(
        "INSERT INTO users (username, full_name, password, role, is_active) "
        "VALUES (%s,%s,%s,'viewer',0)",
        (username, full_name, hash_password(password)),
        fetch=False,
    )
    return (True, "Registrasi berhasil! Tunggu aktivasi Admin.") if ok else (False, "Gagal mendaftar.")


# ─── INIT DB ─────────────────────────────────────────────────────────────────
if "db_initialized" not in st.session_state:
    with st.spinner("Menginisialisasi database..."):
        init_db()
    st.session_state["db_initialized"] = True

# ─── AUTO-LOGIN ──────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    user_data = load_from_cookie(cookies)
    if user_data:
        set_session(user_data)
        st.switch_page("pages/1_Home.py")

if st.session_state.get("logged_in"):
    st.switch_page("pages/1_Home.py")

# ─── UI LOGIN ────────────────────────────────────────────────────────────────
st.markdown("<div class='auth-wrap'>", unsafe_allow_html=True)
st.markdown("""
<div class='auth-card'>
  <div class='brand-logo'>🏝️</div>
  <div class='brand-title'>Villas R Us</div>
  <div class='brand-sub'>PT BALI CIPTA VILA MANDIRI · ANALYTICS DASHBOARD</div>
</div>
""", unsafe_allow_html=True)

tab_login, tab_register = st.tabs(["🔐 Login", "📝 Daftar Akun"])

with tab_login:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    username = st.text_input("Username", placeholder="Masukkan username", key="login_user")
    password = st.text_input("Password", type="password", placeholder="Masukkan password", key="login_pass")

    if st.button("🔐  Masuk ke Dashboard", key="btn_login"):
        if not username or not password:
            st.error("Username dan password wajib diisi.")
    else:
        with st.spinner("Memverifikasi..."):
            user = get_user_by_credentials(username.strip(), password)
        if user:
            set_session(user)
            save_to_cookie(user, cookies)
            # ✅ Fix #2: rerun dulu agar cookie tersimpan, BARU switch_page
            st.rerun()
        else:
            st.error("❌ Username atau password salah, atau akun belum diaktifkan.")

    st.markdown("""
    <div class='hint-box'>
      💡 <b>Default:</b> username <code>admin</code> · password <code>admin123</code>
    </div>""", unsafe_allow_html=True)

with tab_register:
    with st.form("form_register", clear_on_submit=True):
        reg_fullname = st.text_input("Nama Lengkap *", key="reg_fname")
        reg_username = st.text_input("Username *", key="reg_uname")
        reg_password = st.text_input("Password *", type="password", key="reg_pw1")
        reg_password2 = st.text_input("Konfirmasi Password *", type="password", key="reg_pw2")

        if st.form_submit_button("📝  Daftar Sekarang", use_container_width=True):
            if not all([reg_fullname, reg_username, reg_password, reg_password2]):
                st.error("Semua field wajib diisi.")
            elif " " in reg_username:
                st.error("Username tidak boleh mengandung spasi.")
            elif len(reg_password) < 6:
                st.error("Password minimal 6 karakter.")
            elif reg_password != reg_password2:
                st.error("Konfirmasi password tidak cocok.")
            else:
                ok, msg = register_user(
                    reg_username.strip().lower(), reg_fullname.strip(), reg_password
                )
                (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")

    st.markdown("""
    <div class='reg-info'>
      ℹ️ Akun baru perlu <b>diaktifkan oleh Admin</b> sebelum bisa login.
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)