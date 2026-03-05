import json
import datetime
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

COOKIE_KEY = "auth_data"
COOKIE_EXPIRE_HOURS = 24        # ← ganti dari hari ke jam
COOKIE_PASSWORD = "vru-secret-key-ganti-ini-32chars!!"
LOGIN_PAGE = "streamlit_app.py"  # ← satu tempat, gampang ganti


def get_cookie_manager():
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
    st.session_state["do_logout"] = False


def save_to_cookie(user: dict, cookies):
    cookie_data = {
        "user_id": user.get("id") or user.get("user_id"),
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user["role"],
        "expires": (
            datetime.datetime.now()
            + datetime.timedelta(hours=COOKIE_EXPIRE_HOURS)  # ← 24 jam
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
            try:
                del cookies[COOKIE_KEY]
                cookies.save()
            except Exception:
                pass
            return None

        return cookie_data
    except Exception:
        return None

def logout(cookies):
    # 1. Hapus cookie DULU sebelum clear session
    try:
        if COOKIE_KEY in cookies:
            del cookies[COOKIE_KEY]
            cookies.save()
    except Exception:
        pass

    # 2. Clear semua session state
    keys_to_delete = [k for k in st.session_state.keys()]
    for key in keys_to_delete:
        del st.session_state[key]

    # 3. Set flag redirect — JANGAN set do_logout lagi
    st.session_state["logged_in"] = False
    st.session_state["do_logout"] = False