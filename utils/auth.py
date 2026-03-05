import json
import datetime
import streamlit as st

COOKIE_KEY  = "vru_auth"
EXPIRE_HOURS = 24
LOGIN_PAGE   = "streamlit_app.py"

# ── lazy import agar tidak crash sebelum patch ────────────────────────────────
def _get_manager():
    import extra_streamlit_components as stx
    return stx.CookieManager(key="vru_cookie_mgr")


def get_cookie_manager():
    """Kembalikan cookie manager. Panggil di paling atas setiap page."""
    return _get_manager()


# extra-streamlit-components tidak punya .ready() — selalu return True
class _FakeReady:
    def ready(self):
        return True

def get_cookie_manager():
    mgr = _get_manager()
    mgr.ready = lambda: True   # compat shim
    return mgr


def set_session(user_data: dict):
    st.session_state["logged_in"]  = True
    st.session_state["user_id"]    = user_data.get("user_id") or user_data.get("id")
    st.session_state["username"]   = user_data["username"]
    st.session_state["full_name"]  = user_data["full_name"]
    st.session_state["role"]       = user_data["role"]
    st.session_state["do_logout"]  = False


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
        cookies.set(COOKIE_KEY, json.dumps(payload),
                    expires_at=datetime.datetime.now() + datetime.timedelta(hours=EXPIRE_HOURS))
    except Exception:
        pass


def load_from_cookie(cookies) -> dict | None:
    try:
        raw = cookies.get(COOKIE_KEY)
        if not raw:
            return None
        data    = json.loads(raw)
        expires = datetime.datetime.fromisoformat(data["expires"])
        if datetime.datetime.now() > expires:
            _delete_cookie(cookies)
            return None
        return data
    except Exception:
        return None


def _delete_cookie(cookies):
    try:
        cookies.delete(COOKIE_KEY)
    except Exception:
        pass


def logout(cookies):
    """Hapus cookie + bersihkan session, lalu redirect ke login."""
    _delete_cookie(cookies)

    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]

    st.session_state["logged_in"] = False
    st.session_state["do_logout"] = False
    st.switch_page(LOGIN_PAGE)