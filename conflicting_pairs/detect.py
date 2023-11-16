import json
import datasets
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import CrossEncoder


DATA_DIR = Path("./report_data")
FILE = DATA_DIR / "yahoo_health_medical_short_classified.json"


def load_data(file):
    with open(file) as f:
        data = json.load(f)
    return data

def extract_perspective(answer, label):
    return [(s, i) for i, (s, persp) in enumerate(answer) if persp == label]

def extract_sentences(q, label):
    all_sentences = []
    for i, answer in enumerate(q["classified_answers"]):
        sentences = extract_perspective(answer, label)
        sentences = [(s, i, j) for s, j in sentences]
        all_sentences.extend(sentences)

    return all_sentences

def extract_conflicting_pairs(all_sentences):
    pairs = []
    num_sentences = len(all_sentences)
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            cause1, answer_idx1, cause_idx1 = all_sentences[i]
            cause2, answer_idx2, cause_idx2 = all_sentences[j]
            score = model.predict([cause1, cause2])
            label = labels[np.argmax(score)]
            if label == "contradiction":
                pair = {
                    "cause1": cause1,
                    "cause2": cause2,
                    "answer_idx1": answer_idx1,
                    "cause_idx1": cause_idx1,
                    "answer_idx2": answer_idx2,
                    "cause_idx2": cause_idx2,
                }
                pairs.append(pair)
    return pairs

labels = ["contradiction", "entailment", "neutral"]

multi_nli = datasets.load_dataset("multi_nli")
model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

data = load_data(FILE)
output_file = DATA_DIR / "yahoo_health_medical_short_classified_contradiction.json"

target_labels = ["cause", "treatment"]

from collections import defaultdict
num_pairs = defaultdict(int)
num_conflicting_pairs = defaultdict(int)
for q in tqdm(data):
    question = q["subject"]
    for label in target_labels:
        all_sentences = extract_sentences(q, label)
        pairs = extract_conflicting_pairs(all_sentences)
        q[f"{label}_contradiction_pairs"] = pairs
        num_conflicting_pairs[label] += len(pairs)
        num_sentences = len(all_sentences)
        num_pairs[label] += num_sentences * (num_sentences - 1) / 2
        
with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

for label in target_labels:
    print(f"Label: {label}")
    print(f"Number of pairs: {num_pairs[label]}")
    print(f"Number of conflicting pairs: {num_conflicting_pairs[label]}")
    print(f"Percentage of conflicting pairs: {num_conflicting_pairs[label]/num_pairs[label]*100:.2f}%")

