import streamlit as st
from utils.auth import logout


def render_sidebar(cookies):
    with st.sidebar:
        st.markdown("""
        <style>
          [data-testid="stSidebarNav"] { display:none !important; }
          section[data-testid="stSidebar"] .stButton>button {
            background:transparent !important; color:#334155 !important;
            border:1px solid #e2e8f0 !important; box-shadow:none !important;
            border-radius:8px !important; font-weight:600 !important;
          }
          section[data-testid="stSidebar"] .stButton>button:hover {
            background:#f0f6ff !important; border-color:#38bdf8 !important;
            color:#0369a1 !important;
          }

          /* Active page highlight */
          section[data-testid="stSidebar"] .nav-active>button {
            background:#eff6ff !important;
            border-color:#38bdf8 !important;
            color:#0369a1 !important;
          }
        </style>
        <div style='padding:12px 0 20px'>
          <div style='font-size:20px;font-weight:800;color:#0f172a'>🏝️ Villas R Us</div>
          <div style='font-size:11px;color:#94a3b8;margin-top:2px'>Analytics Platform</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:12px 14px;margin-bottom:16px'>
          <div style='font-size:11px;color:#94a3b8;text-transform:uppercase;font-weight:700'>Logged in as</div>
          <div style='font-size:14px;color:#0f172a;font-weight:700'>{st.session_state.get('full_name', 'User')}</div>
          <div style='font-size:11px;color:#0369a1'>{st.session_state.get('role', 'viewer').upper()}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div style='font-size:11px;color:#94a3b8;font-weight:700;
            letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px'>
            Navigation</div>""", unsafe_allow_html=True)

        # ── Detect current page untuk highlight aktif ──────────────────────
        try:
            current_page = st.context.headers.get("Referer", "") or ""
        except Exception:
            current_page = ""

        def is_active(page_slug: str) -> str:
            """Return CSS class string jika halaman ini sedang aktif."""
            return "nav-active" if page_slug in current_page else ""

        # ── Menu Items ────────────────────────────────────────────────────
        # Home
        home_active = is_active("Home") or is_active("home") or (
            not any(p in current_page for p in ["Prediksi","Upload","Users","2_","3_","4_"])
        )
        with st.container():
            if home_active:
                st.markdown("<div class='nav-active'>", unsafe_allow_html=True)
            if st.button("🏠  Dashboard", width='stretch', key="nav_home"):
                st.switch_page("pages/1_Home.py")
            if home_active:
                st.markdown("</div>", unsafe_allow_html=True)

        # Prediksi SARIMA
        if st.button("🔮  Prediksi SARIMA", width='stretch', key="nav_prediksi"):
            st.switch_page("pages/2_Prediksi.py")

        # Upload Data
        if st.button("📁  Upload Data", width='stretch', key="nav_upload"):
            st.switch_page("pages/3_Upload.py")

        # User Management (admin only)
        if st.session_state.get("role") == "admin":
            if st.button("👥  User Management", width='stretch', key="nav_users"):
                st.switch_page("pages/4_Users.py")

        st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0'>",
                    unsafe_allow_html=True)

        if st.button("🚪  Logout", width='stretch', key="nav_logout"):
            logout(cookies)