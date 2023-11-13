import os
import torch
from pathlib import Path

MAIN_DIR = Path(os.getenv("CSC2541_DIR", "."))
DATA_DIR = MAIN_DIR / "data"
EXP_DIR = MAIN_DIR / "experiments"

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)