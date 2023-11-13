import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import DATA_DIR
# from nli_medical_questions import HYPOTHESES_MAP, create_sentence
from preprocessing import load_dataset, split_answer

HYPOTHESES_MAP = None
create_sentence = None

"""
Results:
Total: 278941
medical (152911) & medically_related (168637): iou 0.84, left_iou 0.96, 
medically_related (168637) & medical_related (173491): iou 0.91, left_iou 0.95, 
medical_related (173491) & health (231504): iou 0.75, left_iou 0.97, 
health (231504) & health_related (240607): iou 0.96, left_iou 0.99, 

Indicating that 
medical is a subset of medically_related
medically_related is a subset of medical_related
medical_related is a subset of health
health is a subset of health_related
"""

def print_stats(input_path: Path):
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"Total: {total}")
    keys = list(HYPOTHESES_MAP.keys())
    sorted_keys = sorted(keys, key=lambda x: len(df[df[x]]))
    
    union_mask = df[sorted_keys[0]]
    for i in range(len(sorted_keys) - 1):
        key = sorted_keys[i]
        next_key = sorted_keys[i+1]
        intersection = df[union_mask & df[next_key]]
        left_len = len(df[union_mask])
        union_mask = union_mask | df[next_key]
        union = df[union_mask]
        left_iou = len(intersection) / left_len
        iou = len(intersection) / len(union)
        print(f"{key} ({len(df[df[key]])}) & {next_key} ({len(df[df[next_key]])}): iou {iou:.2f}, left_iou {left_iou:.2f}, ")

def sample_diff(input_path: Path):
    df = pd.read_csv(input_path)
    keys = list(HYPOTHESES_MAP.keys())
    sorted_keys = sorted(keys, key=lambda x: len(df[df[x]]))

    first_key = sorted_keys[0]
    union_mask = df[first_key]
    
    print(f"New in {first_key}:")
    print(df[df[first_key]]["text"].sample(10))
    for i in range(len(sorted_keys) - 1):
        next_key = sorted_keys[i+1]
        
        print(f"New in {next_key}:")
        print(df[df[next_key] & ~union_mask]["text"].sample(10))

        union_mask = union_mask | df[next_key]
    
    print(f"Left:")
    print(df[~union_mask]["text"].sample(10))

def is_valid_answer(answer: str):
    if not isinstance(answer, str):
        return False
    if len(answer) > 1250:
        return False
    return True

def check_correspondence(json_file: Path, csv_file: Path):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.read_csv(csv_file)
    df.sort_values(by="pos", inplace=True)
    
    medical_data = []
    for row, label in zip(data, df["medical"]):
        if label:
            medical_data.append(row)

    print(f"Total medical: {len(medical_data)}")

    medical_short = []
    for row in medical_data:
        subject_len = len(row["subject"]) if isinstance(row["subject"], str) else 0
        content_len = len(row["content"]) if isinstance(row["content"], str) else 0
        if subject_len + content_len > 1250:
            continue
        medical_short.append(row)
    
    print(f"Total medical short: {len(medical_short)}")

    new_data = []
    for row in medical_short:
        if row["nbestanswers"] is None:
            continue
        num_valid_answers = sum([is_valid_answer(a) for a in row["nbestanswers"]])
        if num_valid_answers < 4 or num_valid_answers > 6:
            continue
        new_data.append(row)
        
    print(f"Total medical short valid: {len(new_data)}")

    non_mental_health = []
    for row in new_data:
        if row["cat"] == "Mental Health":
            continue
        non_mental_health.append(row)

    print(f"Total non mental health: {len(non_mental_health)}")

    import random
    random_10k = random.sample(non_mental_health, 5000)
    num_sentences = 0
    for row in tqdm(random_10k):
        num_sentences += sum([len(split_answer(a)) for a in row["nbestanswers"]])
    print(f"Total sentences: {num_sentences}")

def copy_medical_labels(medical_file: Path, target_file: Path, output_file: Path):
    df = pd.read_csv(medical_file)
    target_df = pd.read_csv(target_file)
    df["text"] = df.apply(lambda row: create_sentence(row), axis=1)
    medical_texts = set(df[df["medical"]]["text"])

    print(f"Total: {len(target_df)}")

    target_df["text"] = target_df.apply(lambda row: create_sentence(row), axis=1)
    target_df["medical"] = target_df["text"].apply(lambda x: x in medical_texts)
    print(f"Total medical: {len(target_df[target_df['medical']])}")

    ## remove duplicates
    target_df.drop_duplicates(subset=["subject", "content", "answer", "sentence"], inplace=True)
    print(f"Total medical without dups: {len(target_df[target_df['medical']])}")

    ## count short data
    target_df["text_len"] = target_df["text"].apply(lambda x: len(x))
    target_df["answer_len"] = target_df["answer"].apply(lambda x: len(x))
    mask = (target_df["text_len"] <= 1250) & (target_df["answer_len"] <= 1250) & (target_df["medical"])
    print(f"Total labeled short: {len(target_df[mask])}")

    ## save to file
    target_df.to_csv(output_file, index=False)
       
if __name__ == "__main__":
    file_path = DATA_DIR / "yahoo_health_medical_labels.csv"
    json_file = DATA_DIR / "yahoo_health.json"
    check_correspondence(json_file, file_path)
    # data = load_dataset(json_file)
    # print(len(data))

    # copy_medical_labels(file_path, DATA_DIR / "yahoo_health_labeled_11_02_combined.csv", DATA_DIR / "yahoo_health_labeled_11_07.csv")
    
    