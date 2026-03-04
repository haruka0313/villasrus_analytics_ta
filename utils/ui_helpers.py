# utils/ui_helpers.py
import streamlit as st
from contextlib import contextmanager

@contextmanager
def loading_spinner(message="Memuat data..."):
    """Context manager untuk loading spinner."""
    with st.spinner(message):
        yield

def loading_skeleton():
    """Tampilkan skeleton placeholder saat load."""
    st.markdown("""
    <div style='background:#f1f5f9;border-radius:12px;padding:20px;margin-bottom:12px;
                animation:pulse 1.5s infinite'>
      <div style='background:#e2e8f0;height:16px;border-radius:8px;margin-bottom:8px;width:60%'></div>
      <div style='background:#e2e8f0;height:32px;border-radius:8px;margin-bottom:8px'></div>
      <div style='background:#e2e8f0;height:12px;border-radius:8px;width:40%'></div>
    </div>
    <style>
      @keyframes pulse {
        0%,100% { opacity:1; } 50% { opacity:.5; }
      }
    </style>
    """, unsafe_allow_html=True)