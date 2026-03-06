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
    # ★ FIX UTAMA: Kalau flag logout aktif, JANGAN baca cookie sama sekali.
    # Ini mencegah auto-login balik meski cookie belum diproses browser.
    if st.session_state.get("just_logged_out"):
        return None
    try:
        raw = cookies.get(COOKIE_KEY)
        if not raw or raw.strip() in ("", '""', "null"):
            return None
        data = json.loads(raw)
        # Cookie yang sudah "diracuni" saat logout punya username kosong
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
    try:
        # ★ FIX: "Racuni" cookie dengan payload expired daripada del/"".
        # del dan set-ke-"" tidak bekerja konsisten di EncryptedCookieManager
        # karena value dienkripsi — hasil enkripsi "" != string kosong biasa.
        expired_payload = json.dumps({
            "username": "",
            "expires": "2000-01-01T00:00:00",
        })
        cookies[COOKIE_KEY] = expired_payload
        cookies.save()
    except Exception:
        pass


def logout(cookies):
    # 1. Racuni cookie supaya tidak bisa di-load ulang
    _clear_auth_cookie(cookies)

    # 2. Clear seluruh session state
    st.session_state.clear()

    # 3. Set flag — ini yang memblokir load_from_cookie() di streamlit_app.py
    st.session_state["just_logged_out"] = True

    # 4. Pindah ke login — switch_page lebih aman dari rerun karena
    #    rerun akan re-trigger auth check di halaman aktif yang bisa
    #    masih membaca cookie lama dari memory browser
    st.switch_page("streamlit_app.py")