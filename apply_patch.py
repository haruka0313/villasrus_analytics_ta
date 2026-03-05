import shutil
import site
import os

src = os.path.join(os.path.dirname(__file__), "patches", "encrypted_cookie_manager.py")

for sp in site.getsitepackages():
    dst = os.path.join(sp, "streamlit_cookies_manager", "encrypted_cookie_manager.py")
    if os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Patched: {dst}")
        break