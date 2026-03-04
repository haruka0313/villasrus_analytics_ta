import os
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download, upload_file
from dotenv import load_dotenv

load_dotenv()

def get_config(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

HF_TOKEN = get_config("HF_TOKEN")
HF_REPO  = get_config("HF_REPO")

MODEL_FILES = {
    "briana":   "briana_villas_sarima.pkl",
    "castello": "castello_villas_sarima.pkl",
    "eindra":   "eindra_villas_sarima.pkl",
    "elina":    "elina_villas_sarima.pkl",
    "esha":     "esha_villas_sarima.pkl",
    "isola":    "isola_villas_sarima.pkl",
    "ozamiz":   "ozamiz_villas_sarima.pkl",
}

def load_model(villa_name):
    """Download dari HuggingFace kalau belum ada, lalu load"""
    model_path = f"models/sarima/{MODEL_FILES[villa_name]}"
    os.makedirs("models/sarima", exist_ok=True)

    if not os.path.exists(model_path):
        print(f"⬇️ Downloading {villa_name} model...")
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=MODEL_FILES[villa_name],
            token=HF_TOKEN,
            local_dir="models/sarima"
        )

    return joblib.load(model_path)

def upload_model(villa_name):
    """Upload model ke HuggingFace setelah retrain"""
    model_path = f"models/sarima/{MODEL_FILES[villa_name]}"

    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=MODEL_FILES[villa_name],
        repo_id=HF_REPO,
        token=HF_TOKEN,
        repo_type="model"
    )
    print(f"✅ Model {villa_name} berhasil diupload!")