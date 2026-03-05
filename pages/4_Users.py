import streamlit as st
import hashlib
import pandas as pd
from database import get_conn, run_query

import extra_streamlit_components as stx
from utils.auth import get_cookie_manager, set_session, load_from_cookie, logout
from utils.sidebar import render_sidebar

# ─── STEP 1: Initialize Cookies (MUST BE FIRST) ─────────────────────────────
cookies = get_cookie_manager()
if not cookies.ready():
    st.stop()

# ─── STEP 2: Check Authentication ───────────────────────────────────────────
if not st.session_state.get("logged_in"):
    user_data = load_from_cookie(cookies)
    if user_data:
        set_session(user_data)
    else:
        st.switch_page("app.py")
        st.stop()

# ─── STEP 3: Handle Logout (BEFORE page config & sidebar) ───────────────────
if st.session_state.get("do_logout"):
    logout(cookies)
    st.stop()

# ─── STEP 4: Configure Page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard — Villas R Us",
    page_icon="🏝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STEP 5: Render Sidebar ─────────────────────────────────────────────────
render_sidebar(cookies)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def fmt_date(val, fmt="%d %b %Y", fallback="-"):
    """Safely format a date/datetime — handles None, NaT, and NaN."""
    return val.strftime(fmt) if pd.notna(val) else fallback


def get_all_users():
    return run_query("""
        SELECT id, username, full_name, role, is_active, created_at, last_login
        FROM users ORDER BY created_at DESC
    """)


def get_pending_users():
    return run_query("""
        SELECT id, username, full_name, created_at
        FROM users WHERE is_active=0 ORDER BY created_at DESC
    """)


def create_user(username, full_name, password, role, is_active=1):
    existing = run_query("SELECT id FROM users WHERE username=%s", (username,))
    if existing is not None and not existing.empty:
        return False, f"Username '@{username}' sudah digunakan."
    ok = run_query(
        "INSERT INTO users (username, full_name, password, role, is_active) VALUES (%s,%s,%s,%s,%s)",
        (username, full_name, hash_password(password), role, is_active),
        fetch=False,
    )
    return (True, "User berhasil dibuat.") if ok else (False, "Gagal membuat user.")


def update_user(user_id, full_name, role, is_active):
    ok = run_query(
        "UPDATE users SET full_name=%s, role=%s, is_active=%s WHERE id=%s",
        (full_name, role, is_active, user_id),
        fetch=False,
    )
    return (True, "User berhasil diperbarui.") if ok else (False, "Gagal memperbarui user.")


def activate_user(user_id):
    ok = run_query("UPDATE users SET is_active=1 WHERE id=%s", (user_id,), fetch=False)
    return (True, "User berhasil diaktifkan.") if ok else (False, "Gagal mengaktifkan user.")


def reset_password(user_id, new_password):
    ok = run_query(
        "UPDATE users SET password=%s WHERE id=%s",
        (hash_password(new_password), user_id),
        fetch=False,
    )
    return (True, "Password berhasil direset.") if ok else (False, "Gagal mereset password.")


def delete_user(user_id):
    ok = run_query("DELETE FROM users WHERE id=%s", (user_id,), fetch=False)
    return (True, "User berhasil dihapus.") if ok else (False, "Gagal menghapus user.")


# ─── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family:'Sora',sans-serif; }
  .stApp { background:#f0f6ff; color:#0f172a; }
  #MainMenu, footer { visibility:hidden; }
  [data-testid="stSidebar"] { background:#ffffff !important; border-right:1px solid #e2e8f0; }
  [data-testid="stSidebarNav"] { display:none !important; }

  .section-title {
    font-size:11px; font-weight:700; color:#94a3b8; letter-spacing:.12em;
    text-transform:uppercase; margin-bottom:10px; padding-bottom:6px;
    border-bottom:1px solid #e2e8f0;
  }
  .stat-box {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
    padding:16px 18px; text-align:center; box-shadow:0 2px 8px rgba(3,105,161,0.06);
  }
  .stat-num   { font-size:26px; font-weight:800; font-family:'DM Mono',monospace; }
  .stat-label { font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; }

  .user-card {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
    padding:18px 20px; margin-bottom:12px; box-shadow:0 2px 8px rgba(3,105,161,0.05);
    transition:border-color .2s, box-shadow .2s;
  }
  .user-card:hover { border-color:#38bdf8; box-shadow:0 4px 16px rgba(3,105,161,0.12); }
  .user-card.inactive { opacity:0.55; }

  .pending-card {
    background:#fffbeb; border:1px solid #fde68a; border-radius:12px;
    padding:14px 18px; margin-bottom:10px;
  }

  .role-badge  { display:inline-block; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:700; }
  .role-admin  { background:#dbeafe; color:#1d4ed8; }
  .role-viewer { background:#ede9fe; color:#6d28d9; }

  .status-active   { color:#16a34a; font-size:11px; font-weight:700; }
  .status-inactive { color:#dc2626; font-size:11px; font-weight:700; }
  .status-pending  { color:#d97706; font-size:11px; font-weight:700; }

  input[type="text"], input[type="password"] {
    background:#f8fafc !important; border:1px solid #cbd5e1 !important;
    border-radius:10px !important; color:#0f172a !important;
    -webkit-text-fill-color:#0f172a !important; caret-color:#0369a1 !important;
  }
  input[type="text"]:focus, input[type="password"]:focus {
    border-color:#38bdf8 !important; box-shadow:0 0 0 3px rgba(56,189,248,0.15) !important;
  }
  input::placeholder { color:#94a3b8 !important; -webkit-text-fill-color:#94a3b8 !important; }
  label, .stTextInput label { color:#334155 !important; font-weight:600 !important; font-size:13px !important; }

  .stButton > button { border-radius:8px !important; font-weight:600 !important; font-size:13px !important; }
  div[data-testid="stForm"] {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
    padding:20px; box-shadow:0 2px 8px rgba(3,105,161,0.05);
  }
  .info-panel  { background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:18px 20px; font-size:13px; color:#334155; }
  .warn-panel  { background:#fff7ed; border:1px solid #fed7aa; border-radius:12px; padding:16px 20px; font-size:12px; color:#92400e; }
  .danger-panel{ background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:12px 14px; font-size:12px; color:#991b1b; margin-bottom:10px; }

  section[data-testid="stSidebar"] .stButton > button {
    background:transparent !important; color:#334155 !important;
    border:1px solid #e2e8f0 !important; box-shadow:none !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background:#f0f6ff !important; border-color:#38bdf8 !important; color:#0369a1 !important;
  }
</style>
""", unsafe_allow_html=True)


# ─── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:16px'>
  <span style='font-size:22px;font-weight:800;color:#0f172a'>👥 User Management</span>
  <span style='font-size:13px;color:#94a3b8;margin-left:12px'>Kelola akun &amp; hak akses pengguna</span>
</div>
<hr style='border:none;border-top:1px solid #e2e8f0;margin:0 0 20px'>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────────
with st.spinner("👥 Memuat daftar pengguna..."):
    users_df = get_all_users()
    pending_df = get_pending_users()

users = users_df.to_dict("records")   if users_df   is not None and not users_df.empty   else []
pending = pending_df.to_dict("records") if pending_df is not None and not pending_df.empty else []

current_uid = st.session_state.get("user_id")
ROLES = ["admin", "viewer"]
ROLE_BADGE = {"admin": "role-admin", "viewer": "role-viewer"}

# ─── SUMMARY STATS ──────────────────────────────────────────────────────────────
total = len(users)
active = sum(1 for u in users if u.get("is_active"))
inactive = total - active
admins = sum(1 for u in users if u.get("role") == "admin")
n_pending = len(pending)

c1, c2, c3, c4, c5 = st.columns(5)
for col, num, label, color in [
    (c1, total, "Total User", "#0369a1"),
    (c2, active, "Aktif", "#16a34a"),
    (c3, inactive, "Nonaktif", "#dc2626"),
    (c4, admins, "Admin", "#6d28d9"),
    (c5, n_pending, "Pending", "#d97706"),
]:
    col.markdown(f"""
    <div class='stat-box'>
      <div class='stat-num' style='color:{color}'>{num}</div>
      <div class='stat-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────────────
pending_label = f"⏳ Pending Approval ({n_pending})" if n_pending > 0 else "⏳ Pending Approval"
tab1, tab2, tab3 = st.tabs(["👤 Daftar User", "➕ Tambah User Baru", pending_label])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DAFTAR USER
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Semua Pengguna</div>", unsafe_allow_html=True)

    if not users:
        st.info("Belum ada user terdaftar.")
    else:
        f1, f2, _ = st.columns([2, 2, 4])
        with f1:
            filter_role = st.selectbox("Filter Role", ["Semua"] + ROLES, key="filter_role")
        with f2:
            filter_status = st.selectbox("Filter Status", ["Semua", "Aktif", "Nonaktif"], key="filter_status")

        # Apply filters
        filtered = users.copy()
        if filter_role != "Semua":
            filtered = [u for u in filtered if u.get("role") == filter_role]
        if filter_status == "Aktif":
            filtered = [u for u in filtered if u.get("is_active")]
        elif filter_status == "Nonaktif":
            filtered = [u for u in filtered if not u.get("is_active")]

        st.markdown(
            f"<div style='font-size:12px;color:#94a3b8;margin-bottom:12px'>"
            f"Menampilkan {len(filtered)} dari {total} user</div>",
            unsafe_allow_html=True,
        )

        for u in filtered:
            uid = u["id"]
            uname = u["username"]
            fname = u["full_name"]
            urole = u["role"]
            is_active = bool(u.get("is_active", 1))
            is_self = (uid == current_uid)

            # ── Safe date formatting — pd.notna handles None AND NaT ──────────
            created_str = fmt_date(u.get("created_at"), "%d %b %Y")
            login_str = fmt_date(u.get("last_login"), "%d %b %Y %H:%M", fallback="Belum pernah")

            badge_class = ROLE_BADGE.get(urole, "role-viewer")
            card_class = "user-card" if is_active else "user-card inactive"
            status_html = (
                "<span class='status-active'>● Aktif</span>"
                if is_active else
                "<span class='status-inactive'>● Nonaktif</span>"
            )
            self_label = " <span style='color:#f59e0b;font-size:10px'>(Anda)</span>" if is_self else ""
            icon = "👑" if urole == "admin" else "👁️"

            st.markdown(f"""
            <div class='{card_class}'>
              <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>
                <div style='font-size:28px'>{icon}</div>
                <div style='flex:1'>
                  <div style='font-size:14px;font-weight:700;color:#0f172a'>{fname}{self_label}</div>
                  <div style='font-size:12px;color:#64748b'>@{uname}</div>
                </div>
                <div><span class='role-badge {badge_class}'>{urole.upper()}</span></div>
                <div>{status_html}</div>
                <div style='font-size:11px;color:#94a3b8;text-align:right'>
                  <div>Dibuat: {created_str}</div>
                  <div>Login terakhir: {login_str}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"⚙️ Kelola @{uname}"):
                col_edit, col_pw, col_del = st.columns([3, 3, 2])

                with col_edit:
                    st.markdown("**✏️ Edit Info & Role**")
                    with st.form(key=f"edit_{uid}"):
                        new_fname = st.text_input("Nama Lengkap", value=fname, key=f"fn_{uid}")
                        new_role = st.selectbox(
                            "Role", ROLES,
                            index=ROLES.index(urole) if urole in ROLES else 0,
                            key=f"role_{uid}",
                        )
                        new_active = st.checkbox("Akun Aktif", value=is_active, key=f"act_{uid}", disabled=is_self)
                        if st.form_submit_button("💾 Simpan", width='stretch'):
                            if is_self and not new_active:
                                st.error("Tidak bisa menonaktifkan akun sendiri.")
                            else:
                                with st.spinner("Menyimpan perubahan..."):
                                    ok, msg = update_user(uid, new_fname, new_role, 1 if new_active else 0)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)

                with col_pw:
                    st.markdown("**🔑 Reset Password**")
                    with st.form(key=f"pw_{uid}"):
                        new_pw = st.text_input("Password Baru", type="password", key=f"pw1_{uid}")
                        conf_pw = st.text_input("Konfirmasi", type="password", key=f"pw2_{uid}")
                        if st.form_submit_button("🔄 Reset", width='stretch'):
                            if not new_pw:
                                st.error("Password tidak boleh kosong.")
                            elif len(new_pw) < 6:
                                st.error("Minimal 6 karakter.")
                            elif new_pw != conf_pw:
                                st.error("Konfirmasi tidak cocok.")
                            else:
                                with st.spinner("Mengupdate password..."):
                                    ok, msg = reset_password(uid, new_pw)
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(msg)

                with col_del:
                    st.markdown("**⚠️ Tindakan Lain**")
                    if is_self:
                        st.info("Tidak dapat mengelola akun sendiri.")
                    else:
                        suspend_label = "🔓 Aktifkan" if not is_active else "🔒 Suspend"
                        with st.form(key=f"suspend_{uid}"):
                            if st.form_submit_button(suspend_label, width='stretch'):
                                with st.spinner("Memperbarui status..."):
                                    ok, msg = update_user(uid, fname, urole, 0 if is_active else 1)
                                if ok:
                                    st.rerun()
                                else:
                                    st.error(msg)

                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                        st.markdown("<div class='danger-panel'>⚠️ Hapus permanen.</div>", unsafe_allow_html=True)
                        with st.form(key=f"del_{uid}"):
                            confirm = st.text_input(f"Ketik '{uname}' untuk konfirmasi", key=f"delc_{uid}")
                            if st.form_submit_button("🗑️ Hapus", width='stretch', type="primary"):
                                if confirm.strip() == uname:
                                    with st.spinner("Menghapus user..."):
                                        ok, msg = delete_user(uid)
                                    if ok:
                                        st.success(msg)
                                        st.rerun()
                                    else:
                                        st.error(msg)
                                else:
                                    st.error("Konfirmasi username tidak cocok.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TAMBAH USER
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Buat Akun Pengguna Baru</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])

    with left:
        with st.form("form_add_user", clear_on_submit=True):
            st.markdown("**Informasi Akun**")
            new_username = st.text_input("Username *", placeholder="cth: john_doe")
            new_fullname = st.text_input("Nama Lengkap *", placeholder="cth: John Doe")
            new_role = st.selectbox("Role *", ROLES)
            new_is_active = st.checkbox("Langsung Aktifkan", value=True)
            st.markdown("**Password**")
            new_password = st.text_input("Password *", type="password", placeholder="Min. 6 karakter")
            new_password2 = st.text_input("Konfirmasi Password *", type="password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            if st.form_submit_button("➕ Buat User Baru", width='stretch', type="primary"):
                if not all([new_username, new_fullname, new_password, new_password2]):
                    st.error("Semua field wajib diisi.")
                elif " " in new_username:
                    st.error("Username tidak boleh mengandung spasi.")
                elif len(new_password) < 6:
                    st.error("Password minimal 6 karakter.")
                elif new_password != new_password2:
                    st.error("Konfirmasi password tidak cocok.")
                else:
                    with st.spinner("💾 Membuat akun baru..."):
                        ok, msg = create_user(
                            new_username.strip().lower(),
                            new_fullname.strip(),
                            new_password,
                            new_role,
                            is_active=1 if new_is_active else 0,
                        )
                    if ok:
                        st.success(f"✅ User **@{new_username}** berhasil dibuat!")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    with right:
        st.markdown("""
        <div class='info-panel'>
          <div style='font-weight:700;color:#0369a1;margin-bottom:14px'>📋 Panduan Role</div>
          <div style='margin-bottom:14px'>
            <span style='display:inline-block;padding:3px 12px;border-radius:20px;font-size:11px;
              font-weight:700;background:#dbeafe;color:#1d4ed8;margin-bottom:6px'>ADMIN 👑</span>
            <div style='color:#475569'>Akses penuh: dashboard, upload data, kelola semua user.</div>
          </div>
          <div>
            <span style='display:inline-block;padding:3px 12px;border-radius:20px;font-size:11px;
              font-weight:700;background:#ede9fe;color:#6d28d9;margin-bottom:6px'>VIEWER 👁️</span>
            <div style='color:#475569'>Hanya bisa melihat dashboard &amp; laporan.</div>
          </div>
        </div>
        <div class='warn-panel' style='margin-top:12px'>
          <div style='font-weight:700;margin-bottom:8px'>🔐 Keamanan Password</div>
          <ul style='margin:0;padding-left:16px;line-height:2'>
            <li>Minimal 6 karakter</li>
            <li>Disimpan sebagai hash SHA-256</li>
            <li>Admin dapat reset kapan saja</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — PENDING APPROVAL
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Pendaftaran Menunggu Persetujuan Admin</div>", unsafe_allow_html=True)

    if not pending:
        st.success("✅ Tidak ada pendaftaran yang menunggu approval.")
    else:
        st.markdown(
            f"<div style='font-size:12px;color:#d97706;margin-bottom:12px;font-weight:600'>"
            f"⏳ {len(pending)} user menunggu aktivasi</div>",
            unsafe_allow_html=True,
        )

        for p in pending:
            pid = p["id"]
            puname = p["username"]
            pfname = p["full_name"]

            # ── Safe date formatting ───────────────────────────────────────────
            created_str = fmt_date(p.get("created_at"), "%d %b %Y %H:%M")

            col_info, col_action = st.columns([3, 1])
            with col_info:
                st.markdown(f"""
                <div class='pending-card'>
                  <div style='display:flex;align-items:center;gap:12px'>
                    <div style='font-size:24px'>🕐</div>
                    <div>
                      <div style='font-size:14px;font-weight:700;color:#0f172a'>{pfname}</div>
                      <div style='font-size:12px;color:#64748b'>@{puname}</div>
                      <div style='font-size:11px;color:#94a3b8'>Daftar: {created_str}</div>
                    </div>
                    <span class='status-pending' style='margin-left:auto'>● Pending</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with col_action:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                col_approve, col_reject = st.columns(2)
                with col_approve:
                    if st.button("✅", key=f"approve_{pid}", help=f"Aktifkan @{puname}", width='stretch'):
                        with st.spinner(f"Mengaktifkan @{puname}..."):
                            ok, msg = activate_user(pid)
                        if ok:
                            st.success(f"@{puname} diaktifkan!")
                            st.rerun()
                        else:
                            st.error(msg)
                with col_reject:
                    if st.button("🗑️", key=f"reject_{pid}", help=f"Tolak & hapus @{puname}", width='stretch'):
                        with st.spinner(f"Menolak @{puname}..."):
                            ok, msg = delete_user(pid)
                        if ok:
                            st.warning(f"@{puname} ditolak & dihapus.")
                            st.rerun()
                        else:
                            st.error(msg)
