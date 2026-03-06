import streamlit as st

def render_sidebar(cookies):
    """Render sidebar with navigation and logout button"""

    with st.sidebar:
        # User info header
        st.markdown(f"""
        <div style='padding:20px 0 12px;border-bottom:1px solid #e2e8f0;margin-bottom:16px'>
            <div style='font-size:18px;font-weight:700;color:#0f172a'>
                {st.session_state.get('full_name', 'User')}
            </div>
            <div style='font-size:12px;color:#64748b;margin-top:4px'>
                @{st.session_state.get('username', '')} · {st.session_state.get('role', 'viewer').upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation menu
        st.markdown("<div style='font-size:11px;font-weight:700;color:#94a3b8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px'>Menu</div>", unsafe_allow_html=True)

        if st.button("🏠 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.switch_page("pages/1_Home.py")

        if st.button("🔮 Prediksi SARIMA", use_container_width=True, key="nav_sarima"):
            st.switch_page("pages/2_Prediksi.py")

        if st.button("📤 Upload Data", use_container_width=True, key="nav_upload"):
            st.switch_page("pages/3_Upload.py")

        # Admin-only menu
        if st.session_state.get("role") == "admin":
            if st.button("👥 User Management", use_container_width=True, key="nav_users"):
                st.switch_page("pages/4_Users.py")

        st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0'>", unsafe_allow_html=True)

        # ✅ Logout button - only sets flag, doesn't call logout directly
        if st.button("🚪 Logout", use_container_width=True, key="btn_logout", type="primary"):
            st.session_state["do_logout"] = True
            st.rerun()  # Force immediate rerun to trigger logout handler