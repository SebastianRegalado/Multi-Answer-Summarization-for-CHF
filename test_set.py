import os
import re
import typing as T
from pathlib import Path
import json
import random
import pandas as pd
from preprocessing import split_answer, load_dataset

def choose_test_set(file_path: str, output_path: str, test_size: int):
    random.seed(1234)
    data = load_dataset(file_path)
    chosen_set = set()
    test_set = []
    while len(test_set) < test_size:
        d = random.choice(data)
        answer = random.choice(d["nbestanswers"])
        sentence = random.choice(split_answer(answer))
        if (d["id"], sentence) in chosen_set:
            continue
        chosen_set.add((d["id"], answer))
        row = {
            "subject": d["subject"],
            "content": d["content"],
            "answer": answer,
            "sentence": sentence,
            "cat": d["cat"],
            "maincat": d["maincat"],
            "subcat": d["subcat"],
        }
        test_set.append(row)

    df = pd.DataFrame(test_set)
    df.to_csv(output_path, index=False)

def copy_labels(src_path: str, dest_path: str):
    src_df = pd.read_csv(src_path)
    dest_df = pd.read_csv(dest_path)
    
    labels_to_copy = [l for l in src_df.columns[1:] if "perspective" in l]
    # add labels to dest_df
    for l in labels_to_copy:
        dest_df[l] = None

    for _, row in src_df.iterrows():
        subject, content, sentence, answer = row["subject"], row["content"], row["sentence"], row["answer"]
        mask = (dest_df["subject"] == subject) & (dest_df["sentence"] == sentence) & (dest_df["answer"] == answer)
        if isinstance(content, str):
            mask = mask & (dest_df["content"] == content)
        else:
            mask = mask & (dest_df["content"].isna())

        if mask.sum() == 0:
            continue
        dest_df.loc[mask, labels_to_copy] = row[labels_to_copy].values
    dest_df.to_csv(dest_path, index=False)
    
if __name__ == "__main__":
    DATA_DIR = Path(os.getenv("CSC2541_DIR")) / "data"
    TEST_SIZE = 2000
    file_path = DATA_DIR / "yahoo_health.json"
    output_path = DATA_DIR / "yahoo_health_test.csv"
    choose_test_set(file_path, output_path, TEST_SIZE)

    src_path = DATA_DIR / "Perspective Labeled.csv"
    dest_path = DATA_DIR / "yahoo_health_test.csv"
    copy_labels(src_path, dest_path)

