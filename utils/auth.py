import json
import datetime
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

COOKIE_PREFIX  = "vru_"
COOKIE_KEY     = "auth"
EXPIRE_HOURS   = 24
COOKIE_SECRET  = "4056875343"


def get_cookie_manager():
    cookies = EncryptedCookieManager(
        prefix=COOKIE_PREFIX,
        password=COOKIE_SECRET,
    )
    if not cookies.ready():
        st.stop()
    return cookies


def set_session(user_data: dict):
    st.session_state["logged_in"] = True
    st.session_state["user_id"]   = user_data.get("user_id") or user_data.get("id")
    st.session_state["username"]  = user_data["username"]
    st.session_state["full_name"] = user_data["full_name"]
    st.session_state["role"]      = user_data["role"]


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
    try:
        cookies[COOKIE_KEY] = json.dumps(payload)
        cookies.save()
    except Exception:
        pass


def load_from_cookie(cookies) -> dict | None:
    try:
        raw = cookies.get(COOKIE_KEY)
        if not raw or raw.strip() in ("", '""', "null"):
            return None
        data = json.loads(raw)
        if not data.get("username"):
            return None
        expires = datetime.datetime.fromisoformat(data["expires"])
        if datetime.datetime.now() > expires:
            _clear_auth_cookie(cookies)
            return None
        return data
    except Exception:
        return None


def _clear_auth_cookie(cookies):
    # FIX #1: gunakan del, bukan set ke "" — karena EncryptedCookieManager
    # mengenkripsi value sehingga set ke "" tidak menghasilkan string kosong
    # di browser, dan pengecekan raw.strip() == "" tidak akan pernah cocok.
    try:
        if COOKIE_KEY in cookies:
            del cookies[COOKIE_KEY]
        cookies.save()
    except Exception:
        pass


def logout(cookies):
    # 1. Hapus cookie dengan benar
    _clear_auth_cookie(cookies)

    # 2. Clear session, tapi sisakan flag logout
    st.session_state.clear()
    st.session_state["just_logged_out"] = True

    # FIX #2: Gunakan st.rerun() dulu — bukan langsung switch_page.
    # Ini memberi browser kesempatan memproses penghapusan cookie sebelum
    # pindah halaman, sehingga auto-login tidak langsung aktif kembali.
    # streamlit_app.py akan mendeteksi just_logged_out dan menampilkan login.
    st.rerun()