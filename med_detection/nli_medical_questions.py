import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import DATA_DIR, get_device

device = get_device()

nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
nli_model.to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
HYPOTHESES_MAP = {
    "health": "This is a health question.",
    "medical": "This is a medical question.",
    "medical_related": "This is a medical related question.",
    "medically_related": "This is a medically related question.",
    "health_related": "This is a health related question.",
}
TEXT_COL = "text"
BATCH_SIZE = 1

def create_sentence(row):
    subject = row["subject"]
    content = str(row["content"])
    sentence = subject + (" " + content) if content else ""
    return sentence

def generate_tokenize(hypothesis):
    def tokenize(batch):
        hypotheses = [hypothesis] * len(batch[TEXT_COL])
        tokenized = tokenizer(batch[TEXT_COL], hypotheses, padding=True, truncation=True, max_length=512)
        return tokenized
    return tokenize

def collate_fn(batch):
    pos = torch.tensor([b["pos"] for b in batch], dtype=int).to(device)
    subject = [b["subject"] for b in batch]
    content = [b["content"] for b in batch]
    text = [b["text"] for b in batch]
    input_ids = [torch.tensor(b["input_ids"], dtype=int) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"], dtype=int) for b in batch]

    input_ids = torch.stack(input_ids, dim=0).to(device)
    attention_mask = torch.stack(attention_mask, dim=0).to(device)
    batch = {
        TEXT_COL: text,
        "pos": pos,
        "subject": subject,
        "content": content,
        "tensors": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    }
    return batch

def reduce_dict(i, d, keys):
    d = {k: v for k, v in d.items() if k in keys}
    d["pos"] = i
    return d

def load_data(file_path: Path):
    random.seed(1234)
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    keep_keys = ["subject", "content"]
    data = [reduce_dict(i, d, keep_keys) for i, d in enumerate(data)]
    og_data = pd.DataFrame(data)
    og_data[TEXT_COL] = og_data.apply(lambda row: create_sentence(row), axis=1)
    og_data["len"] = og_data[TEXT_COL].apply(lambda x: len(x))
    og_data.sort_values(by="len", inplace=True, ascending=False)
    return og_data

def convert_to_dataloader(data: Dataset, hypothesis: str):
    dataset = Dataset.from_pandas(data, preserve_index=False)
    dataset = dataset.map(generate_tokenize(hypothesis), batched=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return dataloader

def get_not_medical_indices(dataloader: DataLoader):
    # pose sequence as a NLI premise and label as a hypothesis
    indices = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        logits = nli_model(**batch["tensors"])[0]
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        new_indices = batch["pos"][probs[:,1] < 0.5].tolist()
        indices.extend(new_indices)
    return indices

def predict(filepath: Path, output_path: Path):
    og_data = load_data(filepath)
    for h_name, hypothesis in HYPOTHESES_MAP.items():
        dataloader = convert_to_dataloader(og_data, hypothesis)
        indices = get_not_medical_indices(dataloader)

        ## add h_name column
        og_data[h_name] = True
        og_data.loc[indices, h_name] = False

    og_data.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    file_path = DATA_DIR / "yahoo_health.json"
    output_path = DATA_DIR / "yahoo_health_medical_labels.csv"
    predict(file_path, output_path)

