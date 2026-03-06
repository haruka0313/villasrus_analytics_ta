import json
import datetime
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

COOKIE_PREFIX  = "vru_"
COOKIE_KEY     = "auth"
EXPIRE_HOURS   = 24
LOGIN_PAGE     = "streamlit_app.py"
COOKIE_SECRET  = "ganti_dengan_string_panjang_random_minimal_32_karakter"


def get_cookie_manager():
    cookies = EncryptedCookieManager(
        prefix=COOKIE_PREFIX,
        password=COOKIE_SECRET,
    )
    if not cookies.ready():
        st.stop()  # tunggu render cycle, Streamlit auto re-run
    return cookies


def set_session(user_data: dict):
    st.session_state["logged_in"] = True
    st.session_state["user_id"]   = user_data.get("user_id") or user_data.get("id")
    st.session_state["username"]  = user_data["username"]
    st.session_state["full_name"] = user_data["full_name"]
    st.session_state["role"]      = user_data["role"]
    st.session_state["do_logout"] = False


def save_to_cookie(user: dict, cookies):
    payload = {
        "user_id":   user.get("id") or user.get("user_id"),
        "username":  user["username"],
        "full_name": user["full_name"],
        "role":      user["role"],
        "expires":   (
            datetime.datetime.now() + datetime.timedelta(hours=EXPIRE_HOURS)
        ).isoformat(),
    }
    cookies[COOKIE_KEY] = json.dumps(payload)
    cookies.save()


def _delete_cookie(cookies):
    """
    Overwrite dengan string kosong + save.
    Lebih reliable daripada cookies.delete() di streamlit-cookies-manager.
    """
    try:
        cookies[COOKIE_KEY] = ""
        cookies.save()
    except Exception:
        pass

def logout(cookies):
    """Hapus cookie, bersihkan session, redirect ke login."""
    _delete_cookie(cookies)

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Flag khusus — mencegah auto-login saat cookie belum ter-clear
    st.session_state["logged_in"]    = False
    st.session_state["do_logout"]    = False
    st.session_state["just_logged_out"] = True  # ← tambahkan ini

    st.switch_page(LOGIN_PAGE)


def load_from_cookie(cookies) -> dict | None:
    # Kalau baru saja logout, jangan baca cookie dulu
    if st.session_state.get("just_logged_out"):
        return None

    try:
        raw = cookies.get(COOKIE_KEY)
        if not raw or raw.strip() == "":
            return None
        data    = json.loads(raw)
        expires = datetime.datetime.fromisoformat(data["expires"])
        if datetime.datetime.now() > expires:
            _delete_cookie(cookies)
            return None
        return data
    except Exception:
        return None