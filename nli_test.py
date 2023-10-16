import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import pipeline

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

MAX_LEN = 64
LABELS = ["information", "cause", "treatment", "suggestion", "experience", "clarification"]
pipe = pipeline(model="facebook/bart-large-mnli", device=device)

def is_valid_answers(answers):
    if answers is None:
        return False
    num_answers = len(answers)
    if num_answers < 3 or num_answers > 7:
        return False
    return True
 
class MyDataset(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        
        self.sentences = []
        for d in data:
            answers = d["nbestanswers"]
            if not is_valid_answers(answers):
                continue
            self.sentences.extend(answers)
        self.sentences = self.sentences[:MAX_LEN]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]

file_path = "data/yahoo_health.json"
dataset = MyDataset(file_path)

data = []
for batch_size in [32]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size, candidate_labels = LABELS), total=len(dataset)):
        scores = {
            label: score for label, score in zip(out["labels"], out["scores"])
        }
        sentence_info = {
            "sentence": out["sequence"],
            "pred_label": out["labels"][0],
            "scores": scores
        }
        data.append(sentence_info)

with open("data/perspective_prediction.json", "w") as f:
    json.dump(data, f, indent=4)