import json
import datetime
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

COOKIE_PREFIX      = "vru_"
COOKIE_KEY         = "auth"
COOKIE_LOGOUT_FLAG = "logout_flag"
EXPIRE_HOURS       = 24
LOGIN_PAGE         = "streamlit_app.py"
COOKIE_SECRET      = "4056875343"


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
    try:
        cookies[COOKIE_KEY] = json.dumps(payload)
        cookies.save()
    except Exception:
        pass

def load_from_cookie(cookies) -> dict | None:
    # Cek logout flag PERTAMA, sebelum apapun
    try:
        if cookies.get(COOKIE_LOGOUT_FLAG) == "1":
            cookies[COOKIE_LOGOUT_FLAG] = ""
            cookies[COOKIE_KEY] = ""
            cookies.save()
            return None  # ← stop di sini, jangan lanjut
    except Exception:
        pass

    try:
        raw = cookies.get(COOKIE_KEY)
        if not raw or raw.strip() == "":
            return None
        data    = json.loads(raw)
        expires = datetime.datetime.fromisoformat(data["expires"])
        if datetime.datetime.now() > expires:
            _clear_auth_cookie(cookies)
            return None
        return data
    except Exception:
        return None

def _clear_auth_cookie(cookies):
    try:
        cookies[COOKIE_KEY] = ""
        cookies.save()
    except Exception:
        pass

def logout(cookies):
    # 1. Set flag dulu SEBELUM hapus auth cookie
    try:
        cookies[COOKIE_LOGOUT_FLAG] = "1"
        cookies[COOKIE_KEY] = ""
        cookies.save()
    except Exception:
        pass

    # 2. Clear session
    st.session_state.clear()

    # 3. Langsung switch — TANPA rerun()
    st.switch_page("streamlit_app.py")