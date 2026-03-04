import json
import datetime
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

COOKIE_KEY = "auth_data"
COOKIE_EXPIRE_DAYS = 7
COOKIE_PASSWORD = "vru-secret-key-ganti-ini-32chars!!"


def get_cookie_manager():
    # ✅ Fix #1: Gunakan prefix unik sebagai key, BUKAN simpan di session_state
    # Streamlit sudah handle instance-nya secara internal
    return EncryptedCookieManager(
        prefix="vru_",
        password=COOKIE_PASSWORD,
    )


def set_session(user_data: dict):
    st.session_state["logged_in"] = True
    st.session_state["user_id"] = user_data.get("user_id") or user_data.get("id")
    st.session_state["username"] = user_data["username"]
    st.session_state["full_name"] = user_data["full_name"]
    st.session_state["role"] = user_data["role"]


def save_to_cookie(user: dict, cookies):
    cookie_data = {
        "user_id": user.get("id") or user.get("user_id"),
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"],
        "expires": (
            datetime.datetime.now() + datetime.timedelta(days=COOKIE_EXPIRE_DAYS)
        ).isoformat(),
    }
    cookies[COOKIE_KEY] = json.dumps(cookie_data)
    cookies.save()


def load_from_cookie(cookies) -> dict | None:
    try:
        raw = cookies.get(COOKIE_KEY)
        if not raw:
            return None
        cookie_data = json.loads(raw)
        expires = datetime.datetime.fromisoformat(cookie_data["expires"])
        if datetime.datetime.now() > expires:
            del cookies[COOKIE_KEY]
            cookies.save()
            return None
        return cookie_data
    except Exception:
        return None


def logout(cookies):
    try:
        if COOKIE_KEY in cookies:
            del cookies[COOKIE_KEY]
            cookies.save()
    except Exception:
        pass
    for k in ["logged_in", "user_id", "username", "full_name", "role"]:
        # ✅ Fix: jangan hapus _cookie_manager dari session_state
        st.session_state.pop(k, None)
    st.switch_page("app.py")