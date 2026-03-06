import streamlit as st
from utils.auth import load_from_cookie, set_session


def require_login(cookies):
    """
    Pastikan user sudah login. Kalau belum, redirect ke halaman login.
    Letakkan pemanggilan ini SETELAH get_cookie_manager() dan SEBELUM
    set_page_config() / render konten apapun.
    """
    if st.session_state.get("logged_in"):
        return  # sudah login, lanjut render halaman

    # FIX: Jika ini adalah rerun hasil dari logout(), jangan coba
    # load_from_cookie — langsung redirect supaya browser punya waktu
    # memproses penghapusan cookie di halaman login.
    just_logged_out = st.session_state.pop("just_logged_out", False)
    if just_logged_out:
        st.switch_page("streamlit_app.py")
        st.stop()

    # Coba restore session dari cookie (misal refresh browser)
    user_data = load_from_cookie(cookies)
    if user_data:
        set_session(user_data)
        st.rerun()  # rerun supaya session ter-set sebelum render
    else:
        st.switch_page("streamlit_app.py")
        st.stop()