import streamlit as st
import hashlib
import json
from dotenv import load_dotenv
from database import init_db_once, get_user_by_credentials, run_query
from utils.auth import get_cookie_manager, set_session, save_to_cookie, load_from_cookie

load_dotenv()

# ─── PAGE CONFIG — harus pertama ─────────────────────────────────────────────
st.set_page_config(
    page_title="Villas R Us · Login",
    page_icon="🏝️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── COOKIE MANAGER ──────────────────────────────────────────────────────────
cookies = get_cookie_manager()  # st.stop() otomatis kalau belum ready

# ─── DEBUG SEMENTARA ─────────────────────────────────────────────────────────

st.write("🍪 DEBUG COOKIES:")
try:
    all_cookies = cookies.getAll() if hasattr(cookies, 'getAll') else {}
    st.write(f"logout_flag: `{cookies.get('logout_flag')}`")
    st.write(f"auth cookie: `{cookies.get('auth')}`")
    st.write(f"session logged_in: `{st.session_state.get('logged_in')}`")
except Exception as e:
    st.write(f"Error baca cookie: {e}")
# ─── END DEBUG ────────────────────────────────────────────────────────────────

# ─── INIT DB ─────────────────────────────────────────────────────────────────
init_db_once()

# ─── AUTO-LOGIN ──────────────────────────────────────────────────────────────
# load_from_cookie() sudah handle flag logout di dalamnya
if not st.session_state.get("logged_in"):
    user_data = load_from_cookie(cookies)
    if user_data:
        set_session(user_data)
        st.switch_page("pages/1_Home.py")
        st.stop()

if st.session_state.get("logged_in"):
    st.switch_page("pages/1_Home.py")
    st.stop()


# ─── HELPERS ─────────────────────────────────────────────────────────────────
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


# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background: #f0f6ff !important; }
  .stApp > header { display: none !important; }
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stSidebar"] { display: none !important; }
  .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 420px !important;
  }
  .auth-card { text-align: center; margin-bottom: 16px; }
  .brand-logo  { font-size: 40px; line-height: 1; }
  .brand-title { font-size: 24px; font-weight: 700; color: #0f172a; margin: 6px 0 2px; }
  .brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: .18em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 0;
  }
  .hint-box, .reg-info {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 8px 12px;
    font-size: 12px; color: #1d4ed8; margin-top: 10px;
  }
  .reg-info { background: #fafafa; border-color: #e2e8f0; color: #64748b; }
  div[data-testid="stTextInput"] { margin-bottom: -8px; }
  div[data-testid="stForm"] { border: none; padding: 0; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── UI ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='auth-card'>
  <div class='brand-logo'>🏝️</div>
  <div class='brand-title'>Villas R Us</div>
  <div class='brand-sub'>PT Bali Cipta Vila Mandiri · Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

tab_login, tab_register = st.tabs(["🔐 Login", "📝 Daftar Akun"])

with tab_login:
    with st.form("form_login"):
        username = st.text_input("Username", placeholder="Masukkan username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Masukkan password", key="login_pass")
        submit_login = st.form_submit_button("🔐  Masuk ke Dashboard", use_container_width=True)

        if submit_login:
            if not username or not password:
                st.error("Username dan password wajib diisi.")
            else:
                with st.spinner("Memverifikasi..."):
                    user = get_user_by_credentials(username.strip(), password)
                if user:
                    set_session(user)
                    save_to_cookie(user, cookies)
                    st.success("✅ Login berhasil! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah, atau akun belum diaktifkan.")

    st.markdown("""
    <div class='hint-box'>
      💡 <b>Default:</b> username <code>admin</code> · password <code>admin123</code>
    </div>""", unsafe_allow_html=True)

with tab_register:
    with st.form("form_register", clear_on_submit=True):
        reg_fullname  = st.text_input("Nama Lengkap *", key="reg_fname")
        reg_username  = st.text_input("Username *",     key="reg_uname")
        reg_password  = st.text_input("Password *",     type="password", key="reg_pw1")
        reg_password2 = st.text_input("Konfirmasi Password *", type="password", key="reg_pw2")
        submit_register = st.form_submit_button("📝  Daftar Sekarang", use_container_width=True)

        if submit_register:
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