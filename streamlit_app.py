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

# ... (CSS sama seperti sebelumnya)

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
        st.stop()

if st.session_state.get("logged_in"):
    st.switch_page("pages/1_Home.py")
    st.stop()

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

    with st.form("form_login"):  # ✅ Wrap in form untuk better UX
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
        reg_fullname = st.text_input("Nama Lengkap *", key="reg_fname")
        reg_username = st.text_input("Username *", key="reg_uname")
        reg_password = st.text_input("Password *", type="password", key="reg_pw1")
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

st.markdown("</div>", unsafe_allow_html=True)