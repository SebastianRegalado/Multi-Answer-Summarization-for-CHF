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

labels = ["contradiction", "entailment", "neutral"]

multi_nli = datasets.load_dataset("multi_nli")
model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

data = load_data(FILE)
output_file = DATA_DIR / "yahoo_health_medical_short_classified_contradiction.json"
num_pairs = 0
num_conflicting_pairs = 0
for q in tqdm(data):
    question = q["subject"]
    all_causes = []
    for i, answer in enumerate(q["classified_answers"]):
        causes = extract_perspective(answer, "cause")
        causes = [(s, i, j) for s, j in causes]
        all_causes.extend(causes)

    q["cause_contradiction_pairs"] = []
    num_causes = len(all_causes)
    for i in range(num_causes):
        for j in range(i+1, num_causes):
            num_pairs += 1
            cause1, answer_idx1, cause_idx1 = all_causes[i]
            cause2, answer_idx2, cause_idx2 = all_causes[j]
            if cause1 == cause2:
                continue
            score = model.predict([cause1, cause2])
            label = labels[np.argmax(score)]
            if label == "contradiction":
                num_conflicting_pairs += 1
                pair = {
                    "cause1": cause1,
                    "cause2": cause2,
                    "answer_idx1": answer_idx1,
                    "cause_idx1": cause_idx1,
                    "answer_idx2": answer_idx2,
                    "cause_idx2": cause_idx2,
                    "score": score.tolist()
                }
                q["cause_contradiction_pairs"].append(pair)

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Number of pairs: {num_pairs}")
print(f"Number of conflicting pairs: {num_conflicting_pairs}")
print(f"Percentage of conflicting pairs: {num_conflicting_pairs/num_pairs*100:.2f}%")
