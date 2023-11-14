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

def merge_sentences(answer: list[tuple[str, str]]) -> list[tuple[str, str]]:
    merged = []
    prev_label = None
    prev_sentence = None
    for sentence, label in answer:
        if prev_label is None:
            prev_label = label
            prev_sentence = sentence
        elif prev_label == label:
            prev_sentence += " " + sentence
        else:
            merged.append((prev_sentence, prev_label))
            prev_label = label
            prev_sentence = sentence
    if prev_sentence is not None:
        merged.append((prev_sentence, prev_label))
    return merged
