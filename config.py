import os
import torch
# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Var pour whisper
MODEL_DIR = os.getenv("MODEL_DIR", "models/")
MODEL_NAME = os.getenv("MODEL_NAME", "base")
HF_TOKEN = os.getenv("HF_TOKEN", "my_token")